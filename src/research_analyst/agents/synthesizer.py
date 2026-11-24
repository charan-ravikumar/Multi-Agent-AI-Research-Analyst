from __future__ import annotations

from typing import Any


from research_analyst.agents.base import BaseAgent, StateLike
from research_analyst.config import settings
from research_analyst.grounding.citations import build_evidence, convert_fact_markers, strip_fact_markers
from research_analyst.grounding.gaps import COVERAGE, OPEN_QUESTION, make_item
from research_analyst.llm.json_parse import parse_json
from research_analyst.safety.sanitize import clean_untrusted
from research_analyst.llm import ModelTier
from research_analyst.models import Fact
from research_analyst.models.citation import Citation
from research_analyst.agents.prompts import synthesizer as prompts

# ── prompts ───────────────────────────────────────────────────────────────────


# ── revision-mode prompts (used when critic_objections is non-empty) ──────────


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
    json_mode = True
    # Multi-paragraph JSON for several sections does not fit in the 2048 default
    # (a 2048-token truncation was observed in logs/agent_traces.jsonl).
    max_tokens = 3500

    async def run(self, state: StateLike) -> dict[str, Any]:
        # ── revision mode: critic has objections to address ────────────────────
        if state.get("critic_objections"):
            return await self._run_revision(state)

        # ── original per-sub-question synthesis mode ───────────────────────────
        sub_question = state.get("current_sub_question", "").strip()
        session_id = state["session_id"]
        iteration = state.get("iteration", 0)

        # ── filter facts to this sub-question ─────────────────────────────────
        my_facts: list[Fact] = [
            f for f in state["extracted_facts"]
            if f.sub_question == sub_question
        ]

        # ── promote failed_sub_questions to gaps ───────────────────────────────
        # The Searcher records sub-questions that returned zero web results.
        # We lift them here into unresolved_gaps so the Critic/Writer see them
        # as explicit coverage holes alongside analytical gaps from the LLM.
        failed = list(state.get("failed_sub_questions", []))
        gap_seeds: list[str] = [
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

        gaps = [g for g in gaps if not g.startswith(prompts.SYNTH_FAIL_PREFIX)][:2]       # model gaps: <= 2, speculative
        system_gaps = list(gap_seeds)                                                # detected by the system
        if synthesis.startswith(prompts.SYNTH_FAIL_TEXT):
            system_gaps.append(f'Synthesis generation failed for sub-question "{sub_question}"; facts may need manual review.')
        elif not my_facts and not gap_seeds:
            system_gaps.append(f'No usable evidence could be extracted for sub-question "{sub_question}".')
        all_gaps = system_gaps + gaps
        gap_items = [make_item(sub_question, COVERAGE, g) for g in system_gaps] + \
                    [make_item(sub_question, OPEN_QUESTION, g) for g in gaps]

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
            "gap_items": gap_items,
        }

    # ── revision mode helper ──────────────────────────────────────────────────

    async def _run_revision(self, state: StateLike) -> dict:
        """
        Holistic revision mode: take the assembled draft + critic objections,
        produce a revised draft.  Called when state["critic_objections"] is
        non-empty.  Returns {"revised_draft": [section, ...]}.
        """
        session_id: str = state["session_id"]
        iteration: int = state.get("iteration", 0)
        objections: list[str] = list(state.get("critic_objections", []))
        draft_sections: list[str] = list(state.get("draft", []))

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
            {"role": "system", "content": prompts.REVISION_SYSTEM},
            {
                "role": "user",
                "content": prompts.REVISION_USER.format(
                    n_sections=len(draft_sections),
                    draft_text=draft_text,
                    n_obj=len(objections),
                    objections_block=objections_block,
                    evidence_block=build_evidence(
                        state.get("extracted_facts", []), settings.evidence_block_max_chars),
                ),
            },
        ]

        revised_sections, _err = await self.generate_parsed(
            messages, _parse_revision, state, retry_message=prompts.REVISION_RETRY, step="synth_revision_parse_error",
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
            revision_failed = True
        else:
            revision_failed = False

        self.log.info(
            f"revision complete: {len(revised_sections)} section(s)",
            step="synth_revision_done",
            session_id=session_id,
            iteration=iteration,
        )

        return {"revised_draft": revised_sections, "revision_failed": revision_failed}

    # ── private helpers ───────────────────────────────────────────────────────

    async def _synthesize_with_facts(
        self,
        sub_question: str,
        facts: list[Fact],
        state: StateLike,
    ) -> tuple[str, list[str], list[str]]:
        """Run LLM synthesis when facts are available. Returns (synthesis, contradictions, gaps)."""
        facts_block = "\n".join(
            f"F{i}. [{f.confidence:.0%} confidence] {clean_untrusted(f.content)}  (source: {clean_untrusted(f.source_title, 150)})"
            for i, f in enumerate(facts, 1)
        )
        messages: list[dict] = [
            {"role": "system", "content": prompts.SYSTEM},
            {
                "role": "user",
                "content": prompts.USER.format(
                    sub_question=sub_question,
                    n=len(facts),
                    facts_block=facts_block,
                ),
            },
        ]
        synthesis, contradictions, gaps = await self._call_with_retry(messages, state)
        id_to_url = {f"F{i}": f.source_url for i, f in enumerate(facts, 1)}
        synthesis, dropped = convert_fact_markers(synthesis, id_to_url)
        if dropped:
            self.log.warning(
                f"dropped {dropped} citation(s) to non-existent fact ids",
                step="synth_bad_citation",
                session_id=state["session_id"],
                iteration=state.get("iteration", 0),
            )
        return (
            synthesis,
            [strip_fact_markers(c) for c in contradictions],
            [strip_fact_markers(g) for g in gaps],
        )

    async def _synthesize_empty(
        self,
        sub_question: str,
        gap_seeds: list[str],
        state: StateLike,
    ) -> tuple[str, list[str], list[str]]:
        """Run LLM synthesis when no facts are available. Returns (synthesis, [], gaps)."""
        reasons = "\n".join(f"- {g}" for g in gap_seeds) if gap_seeds else "- Unknown"
        messages: list[dict] = [
            {"role": "system", "content": prompts.SYSTEM},
            {
                "role": "user",
                "content": prompts.EMPTY_USER.format(
                    sub_question=sub_question,
                    reasons=reasons,
                ),
            },
        ]
        return await self._call_with_retry(messages, state)

    async def _call_with_retry(
        self,
        messages: list[dict],
        state: StateLike,
    ) -> tuple[str, list[str], list[str]]:
        """Two-attempt parse loop. Returns (synthesis, contradictions, gaps)."""
        parsed, _err = await self.generate_parsed(
            messages, _parse_synthesis, state, retry_message=prompts.RETRY, step="synth_parse_error",
        )
        if parsed is not None:
            return parsed

        # both attempts failed — return safe fallback
        self.log.error(
            "synthesis failed after 2 attempts; using fallback text",
            step="synth_error",
            session_id=state["session_id"],
            iteration=state.get("iteration", 0),
        )
        return (
            prompts.SYNTH_FAIL_TEXT,
            [],
            [prompts.SYNTH_FAIL_PREFIX + " — facts may need manual review."],
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def _parse_synthesis(raw: str) -> tuple[str, list[str], list[str]]:
    """Parse LLM JSON → (synthesis, contradictions, gaps). Raises on malformed output."""
    data = parse_json(raw)
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


def _build_citations(facts: list[Fact]) -> list[Citation]:
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
            source_type="arxiv" if "arxiv.org/" in f.source_url else "web",
        )
        for f in best.values()
    ]


def _parse_revision(raw: str) -> list[str]:
    """Parse {"revised_sections": [...]} response from the LLM."""
    data = parse_json(raw)
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

