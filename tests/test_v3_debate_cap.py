"""
tests/test_v3_debate_cap.py  —  Verification 3

WORST-CASE test: Critic always returns objections.
The debate loop MUST terminate at exactly round 3 (the cap), never loop
forever, and correctly mark the resolution as forced rather than organic.

SETUP
-----
  - Planner: fixed 2-sub-question plan (no LLM calls)
  - Searcher, Reader, Synthesizer (synthesis mode): normal mock returns
  - Critic: ALWAYS returns one objection (worst case — never satisfied)
  - Synthesizer (revision mode): returns a mock revised draft
  - Writer: returns mock final report

WHAT WE VERIFY
--------------
After app.ainvoke():

  1. debate_round    == 3      (cap hit exactly)
  2. debate_forced   == True   (forced resolution, not organic)
  3. debate_resolved == True   (loop exited cleanly)
  4. Critic was called exactly 3 times (once per round)
  5. critic_objections is non-empty at forced resolution
  6. final_report exists (Writer ran after forced resolution)
  7. routing_history has exactly 3 "debate | round=N" entries

WHY THIS MATTERS
----------------
The cap is a hard safety boundary — the pipeline must never block indefinitely
on a permanently-dissatisfied Critic.  debate_forced=True lets the Writer
disclose unresolved issues to the reader rather than silently omitting them.

Run with:
    .venv/Scripts/python.exe tests/test_v3_debate_cap.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.reader import ReaderAgent
from agents.searcher import SearcherAgent
from agents.synthesizer import SynthesizerAgent
from agents.writer import WriterAgent
from models import ResearchState
from models.fact import Fact
from models.research_plan import ResearchPlan
from models.search_result import SearchResult
from orchestrator.graph import app

# ── Fixed sub-questions ───────────────────────────────────────────────────────

SQ1 = "What is artificial intelligence?"
SQ2 = "How is AI applied in medicine?"
SQ3 = "What are the limitations of AI in clinical settings?"

# ── Mock implementations ──────────────────────────────────────────────────────

async def _mock_planner(self, state: ResearchState) -> dict:
    plan = ResearchPlan(query=state["query"], sub_questions=[SQ1, SQ2, SQ3])
    return {**state, "research_plan": plan, "sub_questions": [SQ1, SQ2, SQ3]}


async def _mock_searcher(self, state: ResearchState) -> dict:
    sq = state.get("current_sub_question", "")
    result = SearchResult(
        sub_question=sq,
        url=f"https://example.com/{sq[:20].replace(' ', '-')}",
        title=f"Mock result for: {sq[:40]}",
        snippet=f"A mock snippet answering: {sq[:60]}",
    )
    return {"search_results": [result], "failed_sub_questions": []}


async def _mock_reader(self, state: ResearchState) -> dict:
    sq = state.get("current_sub_question", "")
    fact = Fact(
        sub_question=sq,
        content=f"Mock fact for: {sq}",
        source_url="https://example.com",
        source_title="Mock Source",
        confidence=0.9,
    )
    return {"extracted_facts": [fact]}


async def _mock_synthesizer(self, state: ResearchState) -> dict:
    """
    Handles both synthesis mode (original) and revision mode.
    Revision mode is triggered when state["critic_objections"] is non-empty.
    """
    if state.get("critic_objections"):
        # Revision mode: return revised_draft (NOT draft)
        return {
            "revised_draft": [
                "[MOCK REVISION] Draft revised in response to critic objections."
            ]
        }
    # Original synthesis mode: return draft section for this sub-question
    sq = state.get("current_sub_question", "")
    return {
        "draft": [f"[MOCK SYNTHESIS] {sq}"],
        "citations": [],
        "contradictions": [],
        "unresolved_gaps": [],
    }


_critic_call_count: int = 0


async def _mock_critic(self, state: ResearchState) -> dict:
    """WORST CASE: always returns one objection — Critic is never satisfied."""
    global _critic_call_count
    _critic_call_count += 1
    return {
        "critic_objections": [
            "Mock objection: this claim is not supported by multiple independent sources"
        ]
    }


async def _mock_writer(self, state: ResearchState) -> dict:
    sections = state.get("revised_draft") or state.get("draft", [])
    return {"final_report": f"[MOCK REPORT] sections={len(sections)}"}


# ── Test ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    global _critic_call_count
    _critic_call_count = 0

    initial = ResearchState(
        session_id=str(uuid.uuid4()),
        query="What is AI and how is it applied in medicine?",
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
        routing_history=[],
        auto_approve_plan=True,  # skip human checkpoint in automated test
    )

    _thread_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    with (
        patch.object(PlannerAgent,    "run", new=_mock_planner),
        patch.object(SearcherAgent,   "run", new=_mock_searcher),
        patch.object(ReaderAgent,     "run", new=_mock_reader),
        patch.object(SynthesizerAgent,"run", new=_mock_synthesizer),
        patch.object(CriticAgent,     "run", new=_mock_critic),
        patch.object(WriterAgent,     "run", new=_mock_writer),
    ):
        result: ResearchState = await app.ainvoke(initial, config=_thread_cfg)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("VERIFICATION 3 — Debate Loop Cap (Worst Case: Critic Never Satisfied)")
    print("=" * 68)

    print(f"\ndebate_round   : {result.get('debate_round', 0)}")
    print(f"debate_resolved: {result.get('debate_resolved', False)}")
    print(f"debate_forced  : {result.get('debate_forced', False)}")
    print(f"critic_calls   : {_critic_call_count}")

    objections = result.get("critic_objections", [])
    print(f"\ncritic_objections at exit ({len(objections)}):")
    for obj in objections:
        print(f"  - {obj}")

    revised = result.get("revised_draft", [])
    print(f"\nrevised_draft sections: {len(revised)}")
    for i, s in enumerate(revised):
        print(f"  [{i}] {s[:80]}")

    routing = result.get("routing_history", [])
    print(f"\nrouting_history ({len(routing)} entries):")
    for i, e in enumerate(routing):
        print(f"  [{i}] {e}")

    print(f"\nfinal_report: {result.get('final_report', '(missing)')[:80]}")

    # ── Assertions ────────────────────────────────────────────────────────────
    print("\n" + "-" * 68)
    print("Assertions:")
    failures: list[str] = []

    # 1. Loop terminated at exactly round 3
    debate_round = result.get("debate_round", 0)
    if debate_round == 3:
        print(f"  PASS \u2713  [1] debate_round = 3 (cap hit exactly, never more)")
    else:
        msg = f"debate_round = {debate_round}, expected 3"
        print(f"  FAIL \u2717  [1] {msg}")
        failures.append(msg)

    # 2. debate_forced = True
    if result.get("debate_forced") is True:
        print(f"  PASS \u2713  [2] debate_forced = True (forced, not organic)")
    else:
        msg = f"debate_forced = {result.get('debate_forced')!r}, expected True"
        print(f"  FAIL \u2717  [2] {msg}")
        failures.append(msg)

    # 3. debate_resolved = True
    if result.get("debate_resolved") is True:
        print(f"  PASS \u2713  [3] debate_resolved = True (loop exited cleanly)")
    else:
        msg = f"debate_resolved = {result.get('debate_resolved')!r}, expected True"
        print(f"  FAIL \u2717  [3] {msg}")
        failures.append(msg)

    # 4. Critic was called exactly 3 times (once per round, no extra)
    if _critic_call_count == 3:
        print(f"  PASS \u2713  [4] Critic called exactly 3 times (one per round)")
    else:
        msg = f"Critic called {_critic_call_count} times, expected 3"
        print(f"  FAIL \u2717  [4] {msg}")
        failures.append(msg)

    # 5. critic_objections non-empty at forced resolution (critic never satisfied)
    if len(objections) > 0:
        print(f"  PASS \u2713  [5] critic_objections non-empty ({len(objections)}) at forced resolution")
    else:
        msg = "critic_objections is empty but should be non-empty at forced resolution"
        print(f"  FAIL \u2717  [5] {msg}")
        failures.append(msg)

    # 6. final_report exists (Writer ran after forced resolution)
    if result.get("final_report"):
        print(f"  PASS \u2713  [6] final_report exists (Writer ran after forced resolution)")
    else:
        msg = "final_report is empty — Writer did not run"
        print(f"  FAIL \u2717  [6] {msg}")
        failures.append(msg)

    # 7. routing_history has exactly 3 debate round entries
    round_entries = [
        e for e in routing
        if e.startswith("debate | round=") and "| objections=" in e
    ]
    if len(round_entries) == 3:
        print(f"  PASS \u2713  [7] routing_history has 3 debate round entries")
    else:
        msg = f"routing_history has {len(round_entries)} debate round entries, expected 3"
        print(f"  FAIL \u2717  [7] {msg}")
        failures.append(msg)

    print("\n" + "=" * 68)
    if not failures:
        print("ALL ASSERTIONS PASSED \u2713")
        print()
        print("Conclusion: the debate loop terminates at exactly round 3 when the")
        print("Critic always returns objections.  debate_forced=True marks the forced")
        print("resolution, and the Writer receives it to disclose unresolved issues.")
        print("The pipeline never loops forever regardless of Critic behaviour.")
    else:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  \u2022 {f}")
    print("=" * 68 + "\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
