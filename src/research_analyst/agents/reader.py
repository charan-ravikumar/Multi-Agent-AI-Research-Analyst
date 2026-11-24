from __future__ import annotations

import asyncio
from typing import Any


from research_analyst.agents.base import BaseAgent, StateLike
from research_analyst.config import settings
from research_analyst.llm.json_parse import parse_json
from research_analyst.safety.sanitize import clean_untrusted, safe_url
from research_analyst.llm import ModelTier
from research_analyst.models import Fact
from research_analyst.models.search_result import SearchResult
from research_analyst.agents.prompts import reader as prompts

# ── prompts ───────────────────────────────────────────────────────────────────


class ReaderError(RuntimeError):
    """Raised when the Reader cannot determine which sub-question to process."""


class ReaderAgent(BaseAgent):
    """
    Extracts structured facts from a sub-question's sources in ONE batched LLM call.

    Reads  : state["current_sub_question"], state["search_results"]
    Writes : state["extracted_facts"]  (appended via operator.add reducer)

    Batching (one call per sub-question instead of one per source) cuts the Reader's
    LLM calls from N*R to N — the dominant cost and the cause of the Groq 429
    storms seen in logs/agent_traces.jsonl.  Evidence per source is the fetched
    BM25 passages when available, else the search snippet, capped at
    settings.evidence_chars_per_source; at most settings.reader_max_sources sources
    are sent (fetched pages first).

    An LLM/transport failure propagates (the branch wrapper records it); an
    unparseable answer after one retry yields no facts for the branch.
    """

    name = "reader"
    tier = ModelTier.FAST
    json_mode = True
    max_tokens = 1500

    def __init__(self) -> None:
        super().__init__()
        self._sems: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    @property
    def sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if loop not in self._sems:
            self._sems[loop] = asyncio.Semaphore(settings.reader_max_concurrent_llm_calls)
        return self._sems[loop]

    async def run(self, state: StateLike) -> dict[str, Any]:
        sub_question = state.get("current_sub_question", "").strip()
        if not sub_question:
            raise ReaderError(
                "current_sub_question is missing or empty — "
                "the orchestrator must set it before dispatching the Reader."
            )

        session_id = state["session_id"]
        iteration = state.get("iteration", 0)

        my_results: list[SearchResult] = [
            r for r in state["search_results"] if r.sub_question == sub_question
        ]
        batches = _source_batches(my_results)
        sources = batches[0] if batches else []

        self.log.info(
            f"extracting facts from {len(sources)}/{len(my_results)} source(s) for: {sub_question!r}",
            step="reader_start",
            session_id=session_id,
            iteration=iteration,
        )

        if not sources:
            self.log.warning(
                f"no usable sources for sub-question: {sub_question!r}",
                step="reader_empty",
                session_id=session_id,
                iteration=iteration,
            )
            return {**state, "extracted_facts": []}

        facts = await self._extract(sources, sub_question, state)
        n_used = len(sources)
        if not facts and len(batches) > 1:
            # The first batch held nothing usable (e.g. off-topic pages): try the next-best sources once
            # before declaring the sub-question uncovered.
            self.log.info(
                f"no facts from the first {n_used} source(s); trying the next {len(batches[1])}",
                step="reader_next_batch",
                session_id=session_id,
                iteration=iteration,
            )
            facts = await self._extract(batches[1], sub_question, state)
            n_used += len(batches[1])

        self.log.info(
            f"extracted {len(facts)} fact(s) from {n_used} source(s)",
            step="reader_done",
            session_id=session_id,
            iteration=iteration,
        )
        return {**state, "extracted_facts": facts}

    # ── private helpers ───────────────────────────────────────────────────────

    async def _extract(
        self, sources: list[SearchResult], sub_question: str, state: StateLike
    ) -> list[Fact]:
        messages: list[dict] = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.USER.format(
                sub_question=sub_question, sources_block=_sources_block(sources))},
        ]
        facts, _err = await self.generate_parsed(
            messages, lambda raw: _parse_and_build_facts(raw, sub_question, sources, self.log, state), state,
            retry_message=prompts.RETRY, step="reader_parse_error", guard=self.sem,   # sem: rate control across branches
        )
        if facts is not None:
            return facts

        self.log.warning(
            f"no facts after 2 failed parse attempts for: {sub_question!r}",
            step="reader_skip",
            session_id=state["session_id"],
            iteration=state.get("iteration", 0),
        )
        return []


# ── module-level helpers ──────────────────────────────────────────────────────

def _evidence_text(r: SearchResult) -> str:
    text = r.full_text.strip() or r.snippet.strip()
    return text[: settings.evidence_chars_per_source]


def _source_batches(results: list[SearchResult], max_batches: int = 2) -> list[list[SearchResult]]:
    """Usable sources, fetched pages first (richer evidence), split into batches of ``reader_max_sources``."""
    usable = [r for r in results if _evidence_text(r)]
    usable.sort(key=lambda r: 0 if r.full_text.strip() else 1)   # stable: keeps the Searcher's ranking
    k = max(1, settings.reader_max_sources)
    return [usable[i:i + k] for i in range(0, len(usable), k)][:max_batches]


def _sources_block(sources: list[SearchResult]) -> str:
    return "\n\n".join(
        prompts.SOURCE.format(
            i=i,
            title=clean_untrusted(r.title, 200),
            url=safe_url(r.url),
            text=clean_untrusted(_evidence_text(r), settings.evidence_chars_per_source),
        )
        for i, r in enumerate(sources, 1)
    )


def _parse_and_build_facts(
    raw: str,
    sub_question: str,
    sources: list[SearchResult],
    log: Any = None,
    state: Any = None,
) -> list[Fact]:
    """Parse {"facts": [...]} (a bare array is tolerated) into validated Fact objects."""
    data = parse_json(raw)
    if isinstance(data, dict):
        data = data.get("facts", [])
    if not isinstance(data, list):
        raise ValueError(f"Expected {{'facts': [...]}}, got {type(data).__name__}")

    facts: list[Fact] = []
    bad_source = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a JSON object: {item!r}")
        try:
            idx = int(item.get("source", 0))
        except (TypeError, ValueError):
            idx = 0
        if not 1 <= idx <= len(sources):
            bad_source += 1          # can't attribute a claim to a real source -> discard
            continue
        src = sources[idx - 1]
        facts.append(Fact.model_validate({
            "sub_question": sub_question,
            "content": item.get("content", ""),
            "source_url": src.url,
            "source_title": src.title,
            "confidence": item.get("confidence", 0.5),
        }))
    if bad_source and log is not None and state is not None:
        log.warning(
            f"discarded {bad_source} fact(s) with missing/invalid source id",
            step="reader_bad_source",
            session_id=state["session_id"],
            iteration=state.get("iteration", 0),
        )
    return facts
