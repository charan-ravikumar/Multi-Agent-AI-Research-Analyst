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
from research_analyst.retrieval.passages import tokenize
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
    settings.evidence_chars_per_source; at most settings.reader_max_sources sources, read in batches of
    settings.reader_batch_size with early stopping (see _read_until_enough)
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

        facts, stats = await self._read_until_enough(batches, sub_question, state)
        stats["sources_available"] = len(my_results)

        self.log.info(
            f"extracted {len(facts)} fact(s) from {stats['sources_read']} source(s) in {stats['batches']} call(s); "
            f"stopped: {stats['stopped']}",
            step="reader_done",
            session_id=session_id,
            iteration=iteration,
        )
        return {**state, "extracted_facts": facts, "reader_stats": stats}

    async def _read_until_enough(
        self, batches: list[list[SearchResult]], sub_question: str, state: StateLike
    ) -> tuple[list[Fact], dict[str, Any]]:
        """Read sources batch by batch (best sources first) and stop as soon as more reading is unlikely to help.

        Stops when (a) enough distinct facts were collected, or (b) after the first batch, a further batch
        added fewer than ``reader_min_new_facts`` NEW facts (the evidence has saturated).  A first batch that
        yields nothing simply moves on to the next-best sources.  Facts that merely restate an earlier one
        do not count as new.
        """
        collected: list[Fact] = []
        sources_read = calls = 0
        stopped = "no_more_sources"
        for i, batch in enumerate(batches):
            if i > 0:
                self.log.info(f"reading the next {len(batch)} source(s)", step="reader_next_batch",
                              session_id=state["session_id"], iteration=state.get("iteration", 0))
            new = await self._extract(batch, sub_question, state)
            sources_read += len(batch)
            calls += 1
            fresh = novel_facts(new, collected)
            collected.extend(fresh)
            if len(collected) >= settings.reader_target_facts:
                stopped = "enough_facts"
                break
            if i >= 1 and len(fresh) < settings.reader_min_new_facts:
                stopped = "saturated"
                break
        return collected, {"sources_read": sources_read, "batches": calls, "stopped": stopped}

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


def _source_batches(results: list[SearchResult]) -> list[list[SearchResult]]:
    """Usable sources, fetched pages first (richer evidence), capped at ``reader_max_sources`` and split into
    batches of ``reader_batch_size``."""
    usable = [r for r in results if _evidence_text(r)]
    usable.sort(key=lambda r: 0 if r.full_text.strip() else 1)   # stable: keeps the Searcher's ranking
    usable = usable[: max(1, settings.reader_max_sources)]
    k = max(1, settings.reader_batch_size)
    return [usable[i:i + k] for i in range(0, len(usable), k)]


def novel_facts(candidates: list[Fact], existing: list[Fact], threshold: float = 0.7) -> list[Fact]:
    """Facts from ``candidates`` that do not merely restate one in ``existing`` (or an earlier candidate)."""
    seen = [set(tokenize(f.content)) for f in existing]
    out: list[Fact] = []
    for f in candidates:
        toks = set(tokenize(f.content))
        if toks and any(len(toks & s) / len(toks | s) >= threshold for s in seen if s):
            continue
        seen.append(toks)
        out.append(f)
    return out


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
