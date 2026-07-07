from __future__ import annotations

from agents.base import BaseAgent
from config import settings
from llm import ModelTier
from models import ResearchState
from tools.web_search import search_web


class SearcherError(RuntimeError):
    """Raised when the Searcher cannot determine which sub-question to process."""


class SearcherAgent(BaseAgent):
    """
    Fetches web search results for a single sub-question.

    Reads  : state["current_sub_question"]
    Writes : state["search_results"]  (appended via operator.add reducer)
             state["failed_sub_questions"]  (appended when zero results)

    The orchestrator sets current_sub_question before dispatching each
    parallel branch; the operator.add reducer on both output fields merges
    all branches back into the shared state automatically.
    """

    name = "searcher"
    tier = ModelTier.FAST   # high-frequency; fast model keeps costs low

    async def run(self, state: ResearchState) -> ResearchState:
        sub_question = state.get("current_sub_question", "").strip()
        if not sub_question:
            raise SearcherError(
                "current_sub_question is missing or empty — "
                "the orchestrator must set it before dispatching the Searcher."
            )

        session_id = state["session_id"]
        iteration = state.get("iteration", 0)

        self.log.info(
            f"searching for sub-question: {sub_question!r}",
            step="search_start",
            session_id=session_id,
            iteration=iteration,
        )

        results = await search_web(
            sub_question,
            sub_question=sub_question,
            max_results=settings.search_max_results,
        )

        if not results:
            self.log.warning(
                f"zero results for sub-question: {sub_question!r}",
                step="search_empty",
                session_id=session_id,
                iteration=iteration,
            )
            return {
                **state,
                "search_results": [],
                "failed_sub_questions": [sub_question],
            }

        self.log.info(
            f"found {len(results)} result(s) for sub-question: {sub_question!r}",
            step="search_done",
            session_id=session_id,
            iteration=iteration,
        )

        return {
            **state,
            "search_results": results,
            "failed_sub_questions": [],
        }
