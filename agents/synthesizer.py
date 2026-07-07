from __future__ import annotations

import json
import re
from typing import List

from pydantic import ValidationError

from agents.base import BaseAgent
from llm import ModelTier
from models import Fact, ResearchState
from models.citation import Citation

# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a research synthesis assistant.
You will receive a research question and a numbered list of factual claims extracted
from multiple sources. Your task:

1. Write a coherent synthesis paragraph that integrates the facts into a single,
   flowing answer to the research question. Merge overlapping claims; do not repeat
   the same point from multiple sources.
2. Identify any contradictions between sources — cases where two sources make
   mutually incompatible claims. List each as a plain-English string.
3. Identify any gaps — aspects of the research question that the facts do not
   adequately address.

Respond with ONLY a valid JSON object — no markdown fences, no prose:

{
  "synthesis": "<one or more coherent paragraphs answering the research question>",
  "contradictions": ["<contradiction 1>", ...],
  "gaps": ["<gap 1>", ...]
}

Rules:
- If there are no contradictions, return "contradictions": []
- If there are no gaps, return "gaps": []
- The synthesis must be grounded in the provided facts — do not invent new claims.
- Output nothing except the JSON object.\
"""

_USER = """\
Research question: {sub_question}

Facts ({n} total):
{facts_block}

Write a synthesis paragraph, flag any contradictions, and note any gaps.\
"""

_EMPTY_USER = """\
Research question: {sub_question}

No facts were available for this sub-question. The search either returned no results
or the Reader could not extract any verifiable claims from the snippets.

Known reason(s) for missing data:
{reasons}

Produce a synthesis that honestly acknowledges the absence of data.\
"""

_RETRY = (
    "Your previous response could not be parsed or validated.\n"
    "Error: {error}\n\n"
    "Return ONLY the corrected JSON object with no other text."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# ── revision-mode prompts (used when critic_objections is non-empty) ──────────

_REVISION_SYSTEM = """\
You are a research revision assistant.

You will receive a draft report (one or more synthesis sections) and a list of
specific objections from a critic reviewer.  Your task: revise the draft to
address each objection WHILE KEEPING THE REPORT STRICTLY GROUNDED IN THE FACTS
ALREADY PRESENT IN THE DRAFT.  You have no access to additional sources.

━━━ ABSOLUTE PROHIBITION — READ THIS FIRST ━━━
You MUST NOT add any new citation, source, study, report, dataset, journal
reference, journal name, author name, institution name, statistic, or any other
specific piece of evidence that does not already appear verbatim in the current
draft text.  This prohibition is unconditional.  There are no exceptions — even
if a critic objection says a claim "lacks corroboration", "needs a source",
"is unsupported", or "a sub-question is unaddressed".  You do not have access
to any sources beyond what the draft already cites.  Fabricating or guessing
a citation is worse than leaving the claim hedged or removing it.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How to address each objection type:

• Objection: a claim lacks corroboration, relies on a single source, or is
  asserted without citing a source.
  ALLOWED  — Add epistemic hedging language directly to the claim, for example:
               "Based on limited available evidence, X ..."
               "Some initial research suggests X, though independent corroboration
               is not available in the current dataset."
               "Early findings indicate X, but this has not been verified across
               multiple sources in this analysis."
  ALLOWED  — Remove the claim entirely if it cannot be meaningfully hedged and
               its removal does not leave the section incoherent.
  FORBIDDEN — Add, invent, imply, or name any new citation, study, or source.

• Objection: a sub-question is not addressed or is insufficiently covered.
  ALLOWED  — Add a brief, honest disclosure of the gap, for example:
               "The research gathered for this report did not surface sufficient
               evidence on [topic].  This represents a gap in the current
               dataset that a follow-up search could address."
  FORBIDDEN — Write a substantive answer using knowledge not present in the
               draft, or cite any new source to fill the gap.  Do not pretend
               the gap is filled when it is not.

• Objection: two sections or sources contradict each other.
  ALLOWED  — Explicitly flag the disagreement, for example:
               "Sources consulted for this report differ on this point: some
               indicate X, while others suggest Y.  The discrepancy could not
               be resolved from the available evidence."
  FORBIDDEN — Resolve the contradiction by citing a new authoritative source.

• Objection: the draft contains repetition, poor structure, or unclear prose.
  ALLOWED  — Restructure, reorder, merge, or rewrite for clarity.
  FORBIDDEN — Introduce new factual claims while doing so.

General rules:
  - You MAY remove claims that cannot be adequately hedged.  Removing an
    unsupported claim is preferable to fabricating support for it.
  - Maintain the same overall structure (one section per sub-question).
  - Do not introduce factual claims beyond what the draft already contains.

Respond with ONLY a valid JSON object — no markdown fences, no prose:

{"revised_sections": ["<revised section 1>", "<revised section 2>", ...]}

Return the same number of sections as the input unless merging is necessary.
Output nothing except the JSON object.\
"""

_REVISION_USER = """\
Current draft ({n_sections} section(s)):
{draft_text}

Critic objections to address ({n_obj}):
{objections_block}

Revise the draft to address every objection above.
Remember: do NOT add any new citation, source name, study, or institution —
hedge unsupported claims with epistemic language or remove them entirely.\
"""

_REVISION_RETRY = (
    "Your previous response could not be parsed.\n"
    "Error: {error}\n\n"
    "Return ONLY the corrected JSON object with no other text."
)


class SynthesizerAgent(BaseAgent):
    """
    Merges extracted facts into a synthesis paragraph, flags contradictions,
    and records knowledge gaps. Also promotes failed_sub_questions into
    unresolved_gaps so the Critic and Writer see coverage holes explicitly.

    Reads  : state["current_sub_question"], state["extracted_facts"],
             state["failed_sub_questions"]
    Writes : state["draft"]          -- Annotated[List[str], operator.add];
                                        each branch appends ONE synthesis paragraph
             state["citations"]      -- deduplicated source list
             state["contradictions"] -- operator.add list
             state["unresolved_gaps"]-- operator.add list (includes failed searches)
    """

    name = "synthesizer"
    tier = ModelTier.STRONG

    async def run(self, state: ResearchState) -> ResearchState:
        # ── revision mode: critic has objections to address ────────────────────
        if state.get("critic_objections"):
            return await self._run_revision(state)

        # ── original per-sub-question synthesis mode ───────────────────────────
        sub_question = state.get("current_sub_question", "").strip()
        session_id = state["session_id"]
        iteration = state.get("iteration", 0)

        # ── filter facts to this sub-question ─────────────────────────────────
        my_facts: List[Fact] = [
            f for f in state["extracted_facts"]
            if f.sub_question == sub_question
        ]

        # ── promote failed_sub_questions to gaps ───────────────────────────────
        # The Searcher records sub-questions that returned zero web results.
        # We lift them here into unresolved_gaps so the Critic/Writer see them
        # as explicit coverage holes alongside analytical gaps from the LLM.
        failed = list(state.get("failed_sub_questions", []))
        gap_seeds: List[str] = [
            f"No search results were found for sub-question: \"{sq}\""
            for sq in failed
            if sq == sub_question  # only promote gaps relevant to this branch
        ]

        self.log.info(
            f"synthesizing {len(my_facts)} fact(s) for: {sub_question!r}",
            step="synth_start",
            session_id=session_id,
            iteration=iteration,
        )

        # ── build synthesis via LLM ────────────────────────────────────────────
        if my_facts:
            synthesis, contradictions, gaps = await self._synthesize_with_facts(
                sub_question, my_facts, state
            )
        else:
            synthesis, contradictions, gaps = await self._synthesize_empty(
                sub_question, gap_seeds, state
            )

        all_gaps = gap_seeds + gaps   # failed-search gaps first, then analytical

        # ── build citations from unique sources in the facts ──────────────────
        citations = _build_citations(my_facts)

        self.log.info(
            f"synthesis complete: {len(contradictions)} contradiction(s), "
            f"{len(all_gaps)} gap(s), {len(citations)} citation(s)",
            step="synth_done",
            session_id=session_id,
            iteration=iteration,
        )

        return {
            **state,
            "draft": [synthesis],
            "citations": citations,
            "contradictions": contradictions,
            "unresolved_gaps": all_gaps,
        }

    # ── revision mode helper ──────────────────────────────────────────────────

    async def _run_revision(self, state: ResearchState) -> dict:
        """
        Holistic revision mode: take the assembled draft + critic objections,
        produce a revised draft.  Called when state["critic_objections"] is
        non-empty.  Returns {"revised_draft": [section, ...]}.
        """
        session_id: str = state["session_id"]
        iteration: int = state.get("iteration", 0)
        objections: List[str] = list(state.get("critic_objections", []))
        draft_sections: List[str] = list(state.get("draft", []))

        draft_text = "\n\n".join(draft_sections) if draft_sections else "(no draft)"
        objections_block = "\n".join(
            f"{i}. {obj}" for i, obj in enumerate(objections, 1)
        )

        self.log.info(
            f"revision mode: {len(objections)} objection(s) to address",
            step="synth_revision_start",
            session_id=session_id,
            iteration=iteration,
        )

        messages: list[dict] = [
            {"role": "system", "content": _REVISION_SYSTEM},
            {
                "role": "user",
                "content": _REVISION_USER.format(
                    n_sections=len(draft_sections),
                    draft_text=draft_text,
                    n_obj=len(objections),
                    objections_block=objections_block,
                ),
            },
        ]

        revised_sections: List[str] | None = None
        last_raw = ""
        last_error = ""

        for attempt in range(1, 3):
            if attempt == 2:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {"role": "user", "content": _REVISION_RETRY.format(error=last_error)}
                )

            last_raw, _usage = await self.llm_generate(messages, state=state)

            try:
                revised_sections = _parse_revision(last_raw)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                self.log.warning(
                    f"revision parse failed (attempt {attempt}): {last_error}",
                    step="synth_revision_parse_error",
                    session_id=session_id,
                    iteration=iteration,
                )

        if revised_sections is None:
            # Fallback: return original sections unchanged rather than crashing.
            self.log.error(
                "revision failed to parse after 2 attempts; keeping original draft",
                step="synth_revision_error",
                session_id=session_id,
                iteration=iteration,
            )
            revised_sections = draft_sections

        self.log.info(
            f"revision complete: {len(revised_sections)} section(s)",
            step="synth_revision_done",
            session_id=session_id,
            iteration=iteration,
        )

        return {"revised_draft": revised_sections}

    # ── private helpers ───────────────────────────────────────────────────────

    async def _synthesize_with_facts(
        self,
        sub_question: str,
        facts: List[Fact],
        state: ResearchState,
    ) -> tuple[str, List[str], List[str]]:
        """Run LLM synthesis when facts are available. Returns (synthesis, contradictions, gaps)."""
        facts_block = "\n".join(
            f"{i}. [{f.confidence:.0%} confidence] {f.content}  (source: {f.source_title})"
            for i, f in enumerate(facts, 1)
        )
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    sub_question=sub_question,
                    n=len(facts),
                    facts_block=facts_block,
                ),
            },
        ]
        return await self._call_with_retry(messages, state)

    async def _synthesize_empty(
        self,
        sub_question: str,
        gap_seeds: List[str],
        state: ResearchState,
    ) -> tuple[str, List[str], List[str]]:
        """Run LLM synthesis when no facts are available. Returns (synthesis, [], gaps)."""
        reasons = "\n".join(f"- {g}" for g in gap_seeds) if gap_seeds else "- Unknown"
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _EMPTY_USER.format(
                    sub_question=sub_question,
                    reasons=reasons,
                ),
            },
        ]
        return await self._call_with_retry(messages, state)

    async def _call_with_retry(
        self,
        messages: list[dict],
        state: ResearchState,
    ) -> tuple[str, List[str], List[str]]:
        """Two-attempt parse loop. Returns (synthesis, contradictions, gaps)."""
        last_raw = ""
        last_error = ""

        for attempt in range(1, 3):
            if attempt == 2:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {"role": "user", "content": _RETRY.format(error=last_error)}
                )

            last_raw, _usage = await self.llm_generate(messages, state=state)

            try:
                return _parse_synthesis(last_raw)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                self.log.warning(
                    f"synthesis parse failed (attempt {attempt}): {last_error}",
                    step="synth_parse_error",
                    session_id=state["session_id"],
                    iteration=state.get("iteration", 0),
                )

        # both attempts failed — return safe fallback
        self.log.error(
            "synthesis failed after 2 attempts; using fallback text",
            step="synth_error",
            session_id=state["session_id"],
            iteration=state.get("iteration", 0),
        )
        return (
            "Synthesis could not be generated due to a formatting error.",
            [],
            ["Synthesis generation failed — facts may need manual review."],
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _parse_synthesis(raw: str) -> tuple[str, List[str], List[str]]:
    """Parse LLM JSON → (synthesis, contradictions, gaps). Raises on malformed output."""
    data = json.loads(_strip_fences(raw).strip())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    synthesis = str(data.get("synthesis", "")).strip()
    if not synthesis:
        raise ValueError("'synthesis' field is missing or empty")

    contradictions = data.get("contradictions", [])
    if not isinstance(contradictions, list):
        raise ValueError("'contradictions' must be a JSON array")
    contradictions = [str(c).strip() for c in contradictions if str(c).strip()]

    gaps = data.get("gaps", [])
    if not isinstance(gaps, list):
        raise ValueError("'gaps' must be a JSON array")
    gaps = [str(g).strip() for g in gaps if str(g).strip()]

    return synthesis, contradictions, gaps


def _build_citations(facts: List[Fact]) -> List[Citation]:
    """Deduplicate facts by source URL and build Citation objects.
    Uses the highest-confidence fact from each source as the representative snippet.
    """
    best: dict[str, Fact] = {}
    for f in facts:
        existing = best.get(f.source_url)
        if existing is None or f.confidence > existing.confidence:
            best[f.source_url] = f

    return [
        Citation(
            url=f.source_url,
            title=f.source_title,
            snippet=f.content,
            source_type="web",
        )
        for f in best.values()
    ]


def _parse_revision(raw: str) -> List[str]:
    """Parse {"revised_sections": [...]} response from the LLM."""
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    data = json.loads(raw.strip())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    if "revised_sections" not in data:
        raise ValueError("Missing 'revised_sections' key")
    sections = data["revised_sections"]
    if not isinstance(sections, list):
        raise ValueError(f"'revised_sections' must be a list, got {type(sections).__name__}")
    result = [str(s).strip() for s in sections if str(s).strip()]
    if not result:
        raise ValueError("'revised_sections' list is empty")
    return result

