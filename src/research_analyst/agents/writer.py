from __future__ import annotations

import re
from typing import Any

from research_analyst.agents.base import BaseAgent, StateLike
from research_analyst.observability.prompt_store import resolve
from research_analyst.llm import ModelTier
from research_analyst.grounding.gaps import COVERAGE, curate_gaps, gap_texts, make_item
from research_analyst.observability.progress import progress
from research_analyst.safety.sanitize import sanitize_report_markdown
from research_analyst.observability.usage import tracker
from research_analyst.grounding.citations import (
    analyze, cited_numbers, normalize_citation_brackets, number_citations, render_references, report_citation_stats,
    verify_numbered,
)
from research_analyst.models.citation import Citation
from research_analyst.agents.prompts import writer as prompts

# ── section headers the report MUST contain (case-insensitive match) ──────────
_REQUIRED_HEADERS = [
    "executive summary",
    "key findings",
    "detailed analysis",
    "contradictions",
    "knowledge gaps",
]   # "## References" is appended by code from the verified citation list

# ── prompts ───────────────────────────────────────────────────────────────────


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

        contradictions: list[str] = state.get("contradictions", [])
        # Curated gaps: every system-detected coverage problem + a short, deduplicated list of open questions
        # (the raw per-sub-question LLM gap lists were repetitive noise).  Falls back to the flat list.
        gap_items = state.get("gap_items") or []
        gaps: list[str] = gap_texts(curate_gaps(gap_items)) if gap_items else list(state.get("unresolved_gaps", []))
        citations: list[Citation] = state.get("citations", [])

        # Prefer debate-revised draft if it exists; fall back to raw fan-in draft.
        sections = state.get("revised_draft") or state.get("draft", [])
        facts = state.get("extracted_facts", [])
        grounding = analyze(sections, facts)            # computed on [S:key]-marked text
        numbered, refs, unknown_markers = number_citations(sections, citations)
        draft_text = "\n\n".join(numbered)

        # Append forced-resolution disclosure to gaps so the report discloses it.
        debate_forced: bool = state.get("debate_forced", False)
        forced_objections: list[str] = state.get("critic_objections", [])
        if debate_forced and forced_objections:
            gaps = list(gaps) + [
                "REVIEW NOTE: The critic-synthesizer debate was terminated by the"
                f" round cap with {len(forced_objections)} unresolved issue(s):"
            ] + [f"  - {obj}" for obj in forced_objections]

        review_notes: list[str] = state.get("review_notes", [])
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
            {"role": "system", "content": resolve("writer-system").text},
            {"role": "user", "content": user_msg},
        ]

        last_report = ""
        last_issues: list[str] = []

        for attempt in range(1, 3):
            if attempt == 2:
                messages.append({"role": "assistant", "content": last_report})
                messages.append({
                    "role": "user",
                    "content": prompts.RETRY.format(issues="\n".join(f"- {i}" for i in last_issues)),
                })

            last_report, _usage = await self.llm_generate(messages, state=state)
            last_report = normalize_citation_brackets(last_report, max_ref=len(refs))   # 【1】 / [1, 2] -> [1][2]

            last_issues = _validate_report(last_report, contradictions, gaps, expect_citations=bool(refs))
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
            **report_citation_stats(body),                      # coverage of the FINAL text (the line above is the draft)
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
        out: dict[str, Any] = {"final_report": report, "references": [r for r in refs if r["n"] in cited],
                               "citation_stats": stats, "usage_summary": usage_summary}
        if refs and not cited:                  # safety net: even the corrective retry produced no markers
            msg = ("The final report text carries no source citations (the writer dropped them); the sources consulted are "
                   "listed below the report. Treat the claims as unverified.")
            out["gap_items"] = [make_item("", COVERAGE, msg)]
            out["unresolved_gaps"] = [msg]
        return out


# ── module-level helpers ──────────────────────────────────────────────────────

def _build_user_message(
    *,
    query: str,
    sub_question: str,
    draft: str,
    contradictions: list[str],
    gaps: list[str],
    refs: list[dict],
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

    return prompts.USER.format(
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
    contradictions: list[str],
    gaps: list[str],
    expect_citations: bool = False,
) -> list[str]:
    """
    Return a list of validation issues. Empty list = report is acceptable.

    Checks:
    1. All required section headers are present (case-insensitive).
    2. If contradictions were provided, the Contradictions section is non-trivially
       present (not just the header line).
    3. If gaps were provided, the Knowledge Gaps section is non-trivially present.
    4. If numbered sources were provided, the report keeps at least one [n] citation marker
       (a real run produced a report whose Writer replaced every [n] with 【n】 / dropped them).
    """
    issues: list[str] = []
    lower = report.lower()

    for header in _REQUIRED_HEADERS:
        if header not in lower:
            issues.append(f"Missing required section: '{header}'")

    if expect_citations and not re.search(r"\[\d+\]", report):
        issues.append("No [n] citation markers found although numbered sources were provided: keep the "
                      "ASCII [n] markers (for example [1] or [2][4]) on every claim, exactly as in the draft")

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

