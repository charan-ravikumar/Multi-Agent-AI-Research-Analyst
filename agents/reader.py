from __future__ import annotations

import asyncio
import json
import re
from typing import List

from pydantic import ValidationError

from agents.base import BaseAgent
from config import settings
from llm import ModelTier
from models import Fact, ResearchState
from models.search_result import SearchResult

# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a fact-extraction assistant.
Read the provided web search snippet and extract 1-3 discrete, self-contained
factual claims that are directly relevant to the research question.

Respond with ONLY a valid JSON array — no markdown fences, no prose.
Each element must match this schema exactly:

[
  {
    "content": "<one factual claim, 1-2 sentences, written as a standalone statement>",
    "confidence": <float 0.0-1.0 reflecting certainty and specificity>
  }
]

Rules:
- Extract only specific, verifiable facts — not vague generalisations.
- If the snippet contains no relevant or verifiable facts, return an empty array: []
- Maximum 3 facts per source. Prefer quality over quantity.
- Do not invent or infer facts not explicitly stated in the snippet.
- Output nothing except the JSON array.\
"""

_USER = """\
Research question: {sub_question}

Source title: {title}
Source URL: {url}
Snippet: {snippet}

Extract factual claims from this snippet that help answer the research question.\
"""

_RETRY = (
    "Your previous response could not be parsed or validated.\n"
    "Error: {error}\n\n"
    "Return ONLY the corrected JSON array with no other text."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ReaderError(RuntimeError):
    """Raised when the Reader cannot determine which sub-question to process."""


class ReaderAgent(BaseAgent):
    """
    Extracts structured facts from search result snippets.

    Reads  : state["current_sub_question"], state["search_results"]
    Writes : state["extracted_facts"]  (appended via operator.add reducer)

    Concurrency: processes all results for a sub-question in parallel, but
    caps simultaneous LLM calls at settings.reader_max_concurrent_llm_calls
    (default 3) via a semaphore. This avoids Groq 429 rate-limit errors
    proactively rather than relying solely on the SDK's reactive backoff.

    Note: works from SearchResult.snippet only — full_text will be populated
    by Playwright scraping once that tool is implemented, which will improve
    extraction quality without requiring any changes here.
    """

    name = "reader"
    tier = ModelTier.FAST   # one LLM call per source; fast model keeps cost low

    def __init__(self) -> None:
        super().__init__()
        # Semaphore is instance-level; limits concurrent llm_generate calls
        # within this agent. When the graph fans out (multiple parallel Reader
        # branches), each instance gets its own semaphore — combine with a
        # module-level semaphore if cross-instance limiting is needed.
        self._sem = asyncio.Semaphore(settings.reader_max_concurrent_llm_calls)

    async def run(self, state: ResearchState) -> ResearchState:
        sub_question = state.get("current_sub_question", "").strip()
        if not sub_question:
            raise ReaderError(
                "current_sub_question is missing or empty — "
                "the orchestrator must set it before dispatching the Reader."
            )

        session_id = state["session_id"]
        iteration = state.get("iteration", 0)

        # ── filter to this sub-question's slice ───────────────────────────────
        my_results: List[SearchResult] = [
            r for r in state["search_results"]
            if r.sub_question == sub_question
        ]

        self.log.info(
            f"extracting facts from {len(my_results)} result(s) for: {sub_question!r}",
            step="reader_start",
            session_id=session_id,
            iteration=iteration,
        )

        if not my_results:
            self.log.warning(
                f"no search results found for sub-question: {sub_question!r}",
                step="reader_empty",
                session_id=session_id,
                iteration=iteration,
            )
            return {**state, "extracted_facts": []}

        # ── extract facts for all results concurrently (semaphore-bounded) ─────
        tasks = [
            self._extract_from_result(result, sub_question, state)
            for result in my_results
        ]
        gathered = await asyncio.gather(*tasks)

        all_facts: List[Fact] = []
        for result, facts in zip(my_results, gathered):
            if facts:
                all_facts.extend(facts)
            else:
                self.log.debug(
                    f"no facts extracted from: {result.url}",
                    step="reader_skip",
                    session_id=session_id,
                    iteration=iteration,
                )

        self.log.info(
            f"extracted {len(all_facts)} fact(s) across {len(my_results)} result(s)",
            step="reader_done",
            session_id=session_id,
            iteration=iteration,
        )

        return {**state, "extracted_facts": all_facts}

    # ── private helpers ───────────────────────────────────────────────────────

    async def _extract_from_result(
        self,
        result: SearchResult,
        sub_question: str,
        state: ResearchState,
    ) -> List[Fact]:
        """
        Run one LLM call (with one retry) to extract facts from a single result.
        Returns an empty list on parse failure after retry — never raises.
        """
        # prefer full_text if scraped, fall back to snippet
        text = result.full_text.strip() if result.full_text.strip() else result.snippet.strip()
        if not text:
            return []

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    sub_question=sub_question,
                    title=result.title,
                    url=result.url,
                    snippet=text,
                ),
            },
        ]

        last_raw = ""
        last_error = ""

        for attempt in range(1, 3):
            if attempt == 2:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {"role": "user", "content": _RETRY.format(error=last_error)}
                )

            async with self._sem:  # proactive rate control: max N concurrent LLM calls
                last_raw, _usage = await self.llm_generate(messages, state=state)

            try:
                facts = _parse_and_build_facts(last_raw, sub_question, result)
                return facts
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                self.log.warning(
                    f"fact parse failed (attempt {attempt}): {last_error}",
                    step="reader_parse_error",
                    session_id=state["session_id"],
                    iteration=state.get("iteration", 0),
                    url=result.url,
                )

        # both attempts failed — skip this result
        self.log.warning(
            f"skipping result after 2 failed parse attempts: {result.url}",
            step="reader_skip",
            session_id=state["session_id"],
            iteration=state.get("iteration", 0),
        )
        return []


# ── module-level helpers ──────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _parse_and_build_facts(
    raw: str,
    sub_question: str,
    result: SearchResult,
) -> List[Fact]:
    """Parse the LLM's JSON array and build validated Fact objects."""
    data = json.loads(_strip_fences(raw).strip())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    facts: List[Fact] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a JSON object: {item!r}")
        # Pydantic validates content non-empty, confidence range, etc.
        fact = Fact.model_validate(
            {
                "sub_question": sub_question,
                "content": item.get("content", ""),
                "source_url": result.url,
                "source_title": result.title,
                "confidence": item.get("confidence", 0.5),
            }
        )
        facts.append(fact)

    return facts

