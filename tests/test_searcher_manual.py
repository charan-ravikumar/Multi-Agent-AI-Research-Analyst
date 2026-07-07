"""
tests/test_searcher_manual.py

Standalone integration test for SearcherAgent.
Calls the real DuckDuckGo API (no mocks). Run from the project root:

    .venv/Scripts/python.exe tests/test_searcher_manual.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.searcher import SearcherAgent
from models import ResearchState


def _make_state(sub_question: str) -> ResearchState:
    return ResearchState(
        session_id=str(uuid.uuid4()),
        query="What is the impact of AI on drug discovery timelines?",
        research_plan=None,
        sub_questions=[sub_question],
        current_sub_question=sub_question,
        search_results=[],
        failed_sub_questions=[],
        extracted_facts=[],
        citations=[],
        contradictions=[],
        unresolved_gaps=[],
        draft=[],
        critique=None,
        final_report="",
        iteration=0,
    )


async def main() -> None:
    sub_question = "How does AI accelerate drug target identification?"
    print(f"\n{'='*65}")
    print(f"Sub-question: {sub_question}")
    print(f"{'='*65}\n")

    state = _make_state(sub_question)
    agent = SearcherAgent()

    result = await agent(state)

    results = result["search_results"]
    failed = result["failed_sub_questions"]

    print(f"search_results  : {len(results)} item(s)")
    print(f"failed_sub_questions: {failed}\n")

    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.title}")
        print(f"     {r.url}")
        print(f"     sub_question field: {r.sub_question!r}")
        print()

    # ── assertions ────────────────────────────────────────────────────────────
    assert isinstance(results, list), "search_results must be a list"
    assert isinstance(failed, list), "failed_sub_questions must be a list"

    if results:
        assert len(failed) == 0, \
            "failed_sub_questions must be empty when results were returned"
        assert all(r.sub_question == sub_question for r in results), \
            "every SearchResult.sub_question must match the queried sub-question"
        assert all(r.url.startswith("http") for r in results), \
            "every result must have an http URL"
        print(f"All assertions passed — {len(results)} results, sub_question tagged correctly.")
    else:
        assert failed == [sub_question], \
            "failed_sub_questions must contain the sub-question when zero results"
        print("Zero results — sub-question correctly recorded in failed_sub_questions.")


if __name__ == "__main__":
    asyncio.run(main())
