from __future__ import annotations

import re
from typing import List, Optional

from agents.base import BaseAgent
from llm import ModelTier
from models import ResearchState
from models.citation import Citation
from models.critique import Critique

# ── section headers the report MUST contain (case-insensitive match) ──────────
_REQUIRED_HEADERS = [
    "executive summary",
    "key findings",
    "detailed analysis",
    "contradictions",
    "knowledge gaps",
    "references",
]

# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a research writer producing a structured analytical report in Markdown.

The report MUST contain exactly these six sections in order, with these exact
level-2 headers (## Header Name):

1. ## Executive Summary
2. ## Key Findings
3. ## Detailed Analysis
4. ## Contradictions and Conflicting Evidence
5. ## Knowledge Gaps and Limitations
6. ## References

Section requirements:
- **Executive Summary**: 2-3 sentences capturing the core answer to the research question.
- **Key Findings**: bullet-point list of the most important facts.
- **Detailed Analysis**: flowing prose based on the provided draft synthesis.
- **Contradictions and Conflicting Evidence**: list every contradiction provided
  explicitly. If none were identified, write exactly: "No contradictions were
  identified across sources."
- **Knowledge Gaps and Limitations**: list every gap provided explicitly. If none,
  write: "No significant gaps were identified."
- **References**: numbered list [1], [2], ... — each entry: title and URL.

CRITICAL (Section 5.9 compliance): every item supplied in the contradictions and
gaps lists must appear in the corresponding sections. Omitting or paraphrasing
them into invisibility is not acceptable.

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

Critique / improvement notes (apply if present):
{critique_block}

Citations to use in References:
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
             state["unresolved_gaps"], state["critique"] (optional)
    Writes : state["final_report"]

    Design note: the Writer generates plain Markdown prose rather than JSON.
    JSON parsing is valuable when output maps to multiple distinct typed fields;
    here final_report is a single string, making JSON an overhead with no benefit.
    Instead, a post-generation validator checks that all required section headers
    are present and that disclosure sections are non-empty, triggering a retry if not.
    """

    name = "writer"
    tier = ModelTier.STRONG

    async def run(self, state: ResearchState) -> ResearchState:
        session_id = state["session_id"]
        iteration = state.get("iteration", 0)
        sub_question = state.get("current_sub_question", state.get("query", ""))

        contradictions: List[str] = state.get("contradictions", [])
        gaps: List[str] = state.get("unresolved_gaps", [])
        citations: List[Citation] = state.get("citations", [])
        critique: Optional[Critique] = state.get("critique")

        # Prefer debate-revised draft if it exists; fall back to raw fan-in draft.
        sections = state.get("revised_draft") or state.get("draft", [])
        draft_text = "\n\n".join(sections)

        # Append forced-resolution disclosure to gaps so the report discloses it.
        debate_forced: bool = state.get("debate_forced", False)
        forced_objections: List[str] = state.get("critic_objections", [])
        if debate_forced and forced_objections:
            gaps = list(gaps) + [
                "REVIEW NOTE: The critic-synthesizer debate was terminated by the"
                f" round cap with {len(forced_objections)} unresolved issue(s):"
            ] + [f"  - {obj}" for obj in forced_objections]

        self.log.info(
            f"writing final report: {len(contradictions)} contradiction(s), "
            f"{len(gaps)} gap(s), {len(citations)} citation(s)",
            step="writer_start",
            session_id=session_id,
            iteration=iteration,
        )

        user_msg = _build_user_message(
            query=state["query"],
            sub_question=sub_question,
            draft=draft_text,
            contradictions=contradictions,
            gaps=gaps,
            citations=citations,
            critique=critique,
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

        return {"final_report": last_report.strip()}


# ── module-level helpers ──────────────────────────────────────────────────────

def _build_user_message(
    *,
    query: str,
    sub_question: str,
    draft: str,
    contradictions: List[str],
    gaps: List[str],
    citations: List[Citation],
    critique: Optional[Critique],
) -> str:
    contra_block = (
        "\n".join(f"- {c}" for c in contradictions)
        if contradictions else "(none)"
    )
    gaps_block = (
        "\n".join(f"- {g}" for g in gaps)
        if gaps else "(none)"
    )
    citations_block = "\n".join(
        f"[{i}] {c.title} — {c.url}"
        for i, c in enumerate(citations, 1)
    ) if citations else "(none)"

    critique_block = "(no critique — write from draft directly)"
    if critique is not None:
        parts = []
        if critique.feedback:
            parts.append(f"Feedback: {critique.feedback}")
        if critique.gaps:
            parts.append("Topics to address:")
            parts.extend(f"  - {g}" for g in critique.gaps)
        if parts:
            critique_block = "\n".join(parts)

    return _USER.format(
        query=query,
        sub_question=sub_question,
        draft=draft or "(no draft available)",
        n_contra=len(contradictions),
        contra_block=contra_block,
        n_gaps=len(gaps),
        gaps_block=gaps_block,
        critique_block=critique_block,
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


def _extract_section(report: str, header_keyword: str) -> str:
    """Extract text between the matching ## header and the next ## header."""
    pattern = re.compile(
        r"##[^#][^\n]*" + re.escape(header_keyword) + r"[^\n]*\n(.*?)(?=\n##|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(report)
    return m.group(1).strip() if m else ""

