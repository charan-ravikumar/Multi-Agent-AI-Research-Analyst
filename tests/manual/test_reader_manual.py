"""
tests/test_reader_manual.py

Standalone integration test for ReaderAgent.
Seeds state with real search results (via search_web), then runs ReaderAgent
against the real Groq API. Run from the project root:

    .venv/Scripts/python.exe tests/test_reader_manual.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.reader import ReaderAgent
from models import ResearchState
from tools.web_search import search_web


async def main() -> None:
    sub_question = "How does AI accelerate drug target identification?"
    session_id = str(uuid.uuid4())

    print(f"\n{'='*65}")
    print(f"Sub-question: {sub_question}")
    print(f"{'='*65}\n")

    # ── Step 1: get real search results (3 results to limit LLM calls) ────────
    print("Fetching search results...")
    search_results = await search_web(
        sub_question,
        sub_question=sub_question,
        max_results=3,
    )
    print(f"Got {len(search_results)} search result(s).\n")

    if not search_results:
        print("No search results — cannot run Reader. Exiting.")
        return

    # ── Step 2: build state with real results ─────────────────────────────────
    state = ResearchState(
        session_id=session_id,
        query="What is the impact of AI on drug discovery timelines?",
        research_plan=None,
        sub_questions=[sub_question],
        current_sub_question=sub_question,
        search_results=search_results,
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

    # ── Step 3: run ReaderAgent ───────────────────────────────────────────────
    print("Running ReaderAgent (one LLM call per search result)...\n")
    agent = ReaderAgent()
    result = await agent(state)

    facts = result["extracted_facts"]
    print(f"{'─'*65}")
    print(f"Extracted {len(facts)} fact(s) total:\n")

    for i, f in enumerate(facts, 1):
        print(f"  Fact {i}:")
        print(f"    content    : {f.content}")
        print(f"    confidence : {f.confidence}")
        print(f"    source     : {f.source_title}")
        print(f"    url        : {f.source_url}")
        print(f"    sub_question: {f.sub_question!r}")
        print()

    # ── assertions ────────────────────────────────────────────────────────────
    assert isinstance(facts, list), "extracted_facts must be a list"

    for i, f in enumerate(facts):
        assert isinstance(f.content, str) and f.content.strip(), \
            f"fact[{i}].content must be a non-empty string"
        assert 0.0 <= f.confidence <= 1.0, \
            f"fact[{i}].confidence {f.confidence} out of [0.0, 1.0]"
        assert f.source_url.startswith("http"), \
            f"fact[{i}].source_url must be a valid URL"
        assert f.sub_question == sub_question, \
            f"fact[{i}].sub_question must match the queried sub-question"

    print(f"All assertions passed — {len(facts)} fact(s) extracted and validated.")


if __name__ == "__main__":
    asyncio.run(main())
