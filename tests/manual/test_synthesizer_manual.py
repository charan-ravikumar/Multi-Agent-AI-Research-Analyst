"""
tests/test_synthesizer_manual.py

Standalone integration test for SynthesizerAgent.
Chains: search_web -> ReaderAgent -> SynthesizerAgent (all real APIs).
Run from the project root:

    .venv/Scripts/python.exe tests/test_synthesizer_manual.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.reader import ReaderAgent
from agents.synthesizer import SynthesizerAgent
from models import ResearchState
from tools.web_search import search_web


async def main() -> None:
    sub_question = "How does AI accelerate drug target identification?"
    session_id = str(uuid.uuid4())

    print(f"\n{'='*65}")
    print(f"Sub-question : {sub_question}")
    print(f"Session      : {session_id[:8]}...")
    print(f"{'='*65}\n")

    # ── Step 1: real search results ───────────────────────────────────────────
    print("Step 1: Fetching search results...")
    search_results = await search_web(sub_question, sub_question=sub_question, max_results=3)
    print(f"  Got {len(search_results)} result(s).\n")

    state: ResearchState = ResearchState(
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

    # ── Step 2: extract facts ─────────────────────────────────────────────────
    print("Step 2: Extracting facts (ReaderAgent)...")
    state = await ReaderAgent()(state)
    print(f"  Extracted {len(state['extracted_facts'])} fact(s).\n")

    # ── Step 3: synthesize ────────────────────────────────────────────────────
    print("Step 3: Synthesizing (SynthesizerAgent)...\n")
    state = await SynthesizerAgent()(state)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"{'─'*65}")
    print("DRAFT SYNTHESIS:")
    print(f"{'─'*65}")
    print(state["draft"])
    print()

    print(f"{'─'*65}")
    print(f"CONTRADICTIONS ({len(state['contradictions'])}):")
    if state["contradictions"]:
        for c in state["contradictions"]:
            print(f"  • {c}")
    else:
        print("  (none flagged)")
    print()

    print(f"{'─'*65}")
    print(f"UNRESOLVED GAPS ({len(state['unresolved_gaps'])}):")
    if state["unresolved_gaps"]:
        for g in state["unresolved_gaps"]:
            print(f"  • {g}")
    else:
        print("  (none flagged)")
    print()

    print(f"{'─'*65}")
    print(f"CITATIONS ({len(state['citations'])}):")
    for i, c in enumerate(state["citations"], 1):
        print(f"  {i}. {c.title}")
        print(f"     {c.url}")
    print()

    # ── assertions ────────────────────────────────────────────────────────────
    assert isinstance(state["draft"], str) and state["draft"].strip(), \
        "draft must be a non-empty string"
    assert isinstance(state["contradictions"], list), \
        "contradictions must be a list"
    assert isinstance(state["unresolved_gaps"], list), \
        "unresolved_gaps must be a list"
    assert isinstance(state["citations"], list), \
        "citations must be a list"

    if state["extracted_facts"]:
        assert state["citations"], \
            "citations must be non-empty when facts were extracted"
        for c in state["citations"]:
            assert c.url.startswith("http"), f"citation URL invalid: {c.url}"

    print("All assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
