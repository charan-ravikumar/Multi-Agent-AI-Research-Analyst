from __future__ import annotations

import re
from typing import Any, List

from agents.base import BaseAgent, StateLike
from llm import ModelTier
from core.gaps import curate_gaps, gap_texts
from core.progress import progress
from core.sanitize import sanitize_report_markdown
from core.usage import tracker
from core.citations import (
    analyze, cited_numbers, number_citations, render_references, verify_numbered,
)
from models.citation import Citation

# ── section headers the report MUST contain (case-insensitive match) ──────────
_REQUIRED_HEADERS = [
    "executive summary",
    "key findings",
    "detailed analysis",
    "contradictions",
    "knowledge gaps",
]   # "## References" is appended by code from the verified citation list

# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a research writer producing a structured analytical report in Markdown.

The report MUST contain exactly these five sections in order, with these exact
level-2 headers (## Header Name):

1. ## Executive Summary
2. ## Key Findings
3. ## Detailed Analysis
4. ## Contradictions and Conflicting Evidence
5. ## Knowledge Gaps and Limitations

Section requirements:
- **Executive Summary**: 2-3 sentences capturing the core answer to the research question.
- **Key Findings**: bullet-point list of the most important facts.
- **Detailed Analysis**: a comprehensive, multi-paragraph, flowing analysis based on the
  provided draft synthesis. You must expand on each point, detail the mechanisms or nuances,
  provide contextual details, and aim to write an exhaustive, elaborate breakdown of the findings.
- **Contradictions and Conflicting Evidence**: list every contradiction provided
  explicitly. If none were identified, write exactly: "No contradictions were
  identified across sources."
- **Knowledge Gaps and Limitations**: list every gap provided explicitly. If none,
  write: "No significant gaps were identified."

CRITICAL (Section 5.9 compliance): every item supplied in the contradictions and
gaps lists must appear in the corresponding sections. Omitting or paraphrasing
them into invisibility is not acceptable.

CITATIONS: the draft synthesis marks claims with numeric citation markers like [1] or
[2][4].  Keep these markers attached to the claims they support (in Key Findings and
Detailed Analysis).  NEVER invent a marker number, never cite a source that is not in
the numbered source list, and do NOT write a References section — it is appended
automatically from the sources you actually cited.

Write in clear, professional prose. Do not wrap the output in code fences.\
"""

_USER = """\
Research question: {query}

Sub-question addressed: {sub_question}

Draft synthesis:
{draft}

Contradictions between sources ({n_contra}):
{contra_block}

Knowledge gaps ({n_gaps}):
{gaps_block}

Numbered sources (for context only — do not reproduce them):
{citations_block}

Write the full research report now.\
"""

_RETRY = """\
Your previous report was missing required sections or disclosures.

Issues found:
{issues}

Rewrite the complete report, ensuring every required section header is present and
every contradiction and gap item is explicitly listed in the appropriate section.\
"""


class WriterAgent(BaseAgent):
    """
    Produces the final polished report from the synthesized draft, citations,
    contradictions, and gaps.

Reads  : state["draft"]  (List[str] — one synthesis section per sub-question),
                 state["citations"], state["contradictions"],
             state["unresolved_gaps"]
    Writes : state["final_report"]

    Design note: the Writer generates plain Markdown prose rather than JSON.
    JSON parsing is valuable when output maps to multiple distinct typed fields;
    here final_report is a single string, making JSON an overhead with no benefit.
    Instead, a post-generation validator checks that all required section headers
    are present and that disclosure sections are non-empty, triggering a retry if not.
    """

    name = "writer"
    tier = ModelTier.STRONG
    budget_exempt = True    # a report must still be produced when the budget is spent
    # The report asks for six sections incl. a long analysis; 2048 tokens was ~93% used
    # for a 5-sub-question run in the logs, so truncation is a real risk.
    max_tokens = 4096

    async def run(self, state: StateLike) -> dict[str, Any]:
        session_id = state["session_id"]
        iteration = state.get("iteration", 0)
        sub_question = state.get("current_sub_question", state.get("query", ""))

        contradictions: List[str] = state.get("contradictions", [])
        # Curated gaps: every system-detected coverage problem + a short, deduplicated list of open questions
        # (the raw per-sub-question LLM gap lists were repetitive noise).  Falls back to the flat list.
        gap_items = state.get("gap_items") or []
        gaps: List[str] = gap_texts(curate_gaps(gap_items)) if gap_items else list(state.get("unresolved_gaps", []))
        citations: List[Citation] = state.get("citations", [])

        # Prefer debate-revised draft if it exists; fall back to raw fan-in draft.
        sections = state.get("revised_draft") or state.get("draft", [])
        facts = state.get("extracted_facts", [])
        grounding = analyze(sections, facts)            # computed on [S:key]-marked text
        numbered, refs, unknown_markers = number_citations(sections, citations)
        draft_text = "\n\n".join(numbered)

        # Append forced-resolution disclosure to gaps so the report discloses it.
        debate_forced: bool = state.get("debate_forced", False)
        forced_objections: List[str] = state.get("critic_objections", [])
        if debate_forced and forced_objections:
            gaps = list(gaps) + [
                "REVIEW NOTE: The critic-synthesizer debate was terminated by the"
                f" round cap with {len(forced_objections)} unresolved issue(s):"
            ] + [f"  - {obj}" for obj in forced_objections]

        review_notes: List[str] = state.get("review_notes", [])
        if review_notes:
            gaps = list(gaps) + ["REVIEWER NOTES (minor, not blocking):"] + [f"  - {n}" for n in review_notes[:6]]

        if state.get("debate_budget_exhausted"):
            gaps = list(gaps) + [
                "REVIEW NOTE: the critic/reviser loop stopped early because the run's token budget was exhausted;"
                " the draft may not have been fully reviewed."
            ]

        self.log.info(
            f"writing final report: {len(contradictions)} contradiction(s), "
            f"{len(gaps)} gap(s), {len(refs)} cited source(s)",
            step="writer_start",
            session_id=session_id,
            iteration=iteration,
        )

        progress.advance(session_id, "writing", 1, "Writing the final report…")
        user_msg = _build_user_message(
            query=state["query"],
            sub_question=sub_question,
            draft=draft_text,
            contradictions=contradictions,
            gaps=gaps,
            refs=refs,
        )

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        last_report = ""
        last_issues: List[str] = []

        for attempt in range(1, 3):
            if attempt == 2:
                messages.append({"role": "assistant", "content": last_report})
                messages.append({
                    "role": "user",
                    "content": _RETRY.format(issues="\n".join(f"- {i}" for i in last_issues)),
                })

            last_report, _usage = await self.llm_generate(messages, state=state)

            last_issues = _validate_report(last_report, contradictions, gaps)
            if not last_issues:
                break

            self.log.warning(
                f"report validation failed (attempt {attempt}): {last_issues}",
                step="writer_validate",
                session_id=session_id,
                iteration=iteration,
            )

        if last_issues:
            self.log.error(
                f"report still has issues after 2 attempts: {last_issues}",
                step="writer_error",
                session_id=session_id,
                iteration=iteration,
            )

        self.log.info(
            "final report written",
            step="writer_done",
            session_id=session_id,
            iteration=iteration,
        )

        progress.advance(session_id, "writing", 1, "Checking citations…")
        # ── deterministic citation verification + references (code, not LLM) ──────
        body, removed = verify_numbered(_strip_llm_references(last_report.strip()), {r["n"] for r in refs})
        cited = cited_numbers(body)
        report = sanitize_report_markdown(f"{body}\n\n{render_references(refs, cited)}")
        stats = {
            **{k: v for k, v in grounding.items() if k != "problems"},
            "unknown_markers": grounding["unknown_markers"] + unknown_markers,
            "invalid_markers_removed": removed,
            "sources_cited": len(cited),
            "sources_consulted": len({c.url for c in citations}),
        }
        self.log.info(
            f"citation check: {stats}",
            step="citation_check",
            session_id=session_id,
            iteration=iteration,
        )
        progress.finish_stage(session_id, "writing", "Finishing up…")
        usage_summary = tracker.summary(session_id)
        self.log.info(
            f"run usage: {usage_summary['total_tokens']} tokens in {usage_summary['calls']} calls",
            step="run_usage",
            session_id=session_id,
            iteration=iteration,
            usage=usage_summary,
        )
        return {"final_report": report, "references": [r for r in refs if r["n"] in cited],
                "citation_stats": stats, "usage_summary": usage_summary}


# ── module-level helpers ──────────────────────────────────────────────────────

def _build_user_message(
    *,
    query: str,
    sub_question: str,
    draft: str,
    contradictions: List[str],
    gaps: List[str],
    refs: List[dict],
) -> str:
    contra_block = (
        "\n".join(f"- {c}" for c in contradictions)
        if contradictions else "(none)"
    )
    gaps_block = (
        "\n".join(f"- {g}" for g in gaps)
        if gaps else "(none)"
    )
    citations_block = (
        "\n".join(f"[{r['n']}] {r['title']}" for r in refs) if refs else "(none)"
    )

    return _USER.format(
        query=query,
        sub_question=sub_question,
        draft=draft or "(no draft available)",
        n_contra=len(contradictions),
        contra_block=contra_block,
        n_gaps=len(gaps),
        gaps_block=gaps_block,
        citations_block=citations_block,
    )


def _validate_report(
    report: str,
    contradictions: List[str],
    gaps: List[str],
) -> List[str]:
    """
    Return a list of validation issues. Empty list = report is acceptable.

    Checks:
    1. All required section headers are present (case-insensitive).
    2. If contradictions were provided, the Contradictions section is non-trivially
       present (not just the header line).
    3. If gaps were provided, the Knowledge Gaps section is non-trivially present.
    """
    issues: List[str] = []
    lower = report.lower()

    for header in _REQUIRED_HEADERS:
        if header not in lower:
            issues.append(f"Missing required section: '{header}'")

    # Section 5.9: disclosures must be non-empty when data exists
    if contradictions:
        section = _extract_section(report, "contradictions")
        if not section or "no contradictions" in section.lower() and len(contradictions) > 0:
            issues.append(
                "Contradictions section is empty or says 'none' but contradictions were provided"
            )

    if gaps:
        section = _extract_section(report, "knowledge gaps")
        if not section or "no significant gaps" in section.lower() and len(gaps) > 0:
            issues.append(
                "Knowledge Gaps section is empty or says 'none' but gaps were provided"
            )

    return issues


def _strip_llm_references(report: str) -> str:
    """Drop any References section the LLM wrote anyway; ours is built from verified sources."""
    return re.sub(r"(?ims)^#{1,3}\s*references\b[^\n]*\n.*?(?=^#{1,3}\s|\Z)", "", report).strip()


def _extract_section(report: str, header_keyword: str) -> str:
    """Extract text between the matching ## header and the next ## header."""
    pattern = re.compile(
        r"##[^#][^\n]*" + re.escape(header_keyword) + r"[^\n]*\n(.*?)(?=\n##|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(report)
    return m.group(1).strip() if m else ""

