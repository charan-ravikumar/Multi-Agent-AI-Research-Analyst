"""
tests/test_planner_manual.py

Standalone integration test for PlannerAgent.
Hits the real LLM API (no mocks). Run from the project root:

    .venv/Scripts/python.exe tests/test_planner_manual.py

Redis note: if Redis is not running, scratchpad persistence will log a warning
but the test continues -- the LLM call and validation path are unaffected.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# ── ensure project root is on sys.path when run directly ─────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from agents.planner import PlannerAgent
from models import ResearchState


def _make_state(query: str) -> ResearchState:
    """Minimal but fully valid ResearchState for the Planner."""
    return ResearchState(
        session_id=str(uuid.uuid4()),
        query=query,
        research_plan=None,
        sub_questions=[],
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
    query = "What is the impact of AI on drug discovery timelines?"
    print(f"\n{'='*65}")
    print(f"Query : {query}")
    print(f"Min sub-questions : {settings.planner_min_sub_questions}")
    print(f"Max sub-questions : {settings.planner_max_sub_questions}")
    print(f"LLM model (strong): {settings.groq_strong_model}")
    print(f"{'='*65}\n")

    state = _make_state(query)
    agent = PlannerAgent()

    print("Calling PlannerAgent (via __call__ — full BaseAgent wrapper)...\n")
    result = await agent(state)

    sub_questions = result["sub_questions"]
    plan = result["research_plan"]

    print("─── Sub-questions returned ───────────────────────────────────")
    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")
    print()

    if plan:
        print(f"Depth          : {plan.depth}")
        print(f"Strategy notes : {plan.strategy_notes}")
    print()

    # ── assertions ────────────────────────────────────────────────────────────
    assert isinstance(sub_questions, list), \
        f"sub_questions should be a list, got {type(sub_questions)}"

    assert len(sub_questions) > 0, \
        "sub_questions must not be empty"

    assert settings.planner_min_sub_questions <= len(sub_questions) <= settings.planner_max_sub_questions, (
        f"Expected between {settings.planner_min_sub_questions} and "
        f"{settings.planner_max_sub_questions} sub-questions, got {len(sub_questions)}"
    )

    for i, q in enumerate(sub_questions):
        assert isinstance(q, str) and q.strip(), \
            f"sub_questions[{i}] is not a non-empty string: {q!r}"

    print(f"✓ All assertions passed — {len(sub_questions)} sub-questions, "
          f"each a non-empty string.")


if __name__ == "__main__":
    asyncio.run(main())
