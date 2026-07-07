"""
tests/test_human_checkpoint.py  —  Verification 4

Tests the human-in-the-loop checkpoint after the Planner, before fan-out.

Uses the verified interrupt() mechanism (see tests/_verify_interrupt.py):
  - First ainvoke() pauses at _plan_approval_node and returns with __interrupt__
  - Command(resume=edited_list) resumes from the pause point
  - Fan-out dispatches ONLY the approved/edited sub_questions

WHAT WE VERIFY
--------------
Part A — Pause:
  1. First ainvoke() returns with result["__interrupt__"] present
  2. The interrupt value contains the planned sub_questions (from mock planner)
  3. The pipeline has NOT proceeded to fan-out yet (no draft sections)

Part B — Edited resume:
  4. Remove one sub_question (SQ2) from the approved list before resuming
  5. After resume, result["sub_questions"] == [SQ1, SQ3] (edited list)
  6. result["draft"] has exactly 2 sections (only SQ1 and SQ3 were fanned out)
  7. No draft section mentions SQ2 (it was dropped by the human review)

Run with:
    .venv/Scripts/python.exe tests/test_human_checkpoint.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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

# ── Fixed sub-questions (planner mock returns all three) ──────────────────────

SQ1 = "What is the role of ML in target identification?"
SQ2 = "How does AI reduce clinical trial failure rates?"   # ← human will remove this
SQ3 = "What are the limitations of AI in drug discovery?"

# ── Mock implementations ──────────────────────────────────────────────────────

async def _mock_planner(self, state: ResearchState) -> dict:
    """Returns a fixed 3-sub-question plan — human review will remove SQ2."""
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
    if state.get("critic_objections"):
        return {"revised_draft": ["[MOCK REVISION]"]}
    sq = state.get("current_sub_question", "")
    return {
        "draft": [f"[MOCK SYNTHESIS for: {sq}]"],
        "citations": [],
        "contradictions": [],
        "unresolved_gaps": [],
    }


async def _mock_critic(self, state: ResearchState) -> dict:
    return {"critic_objections": []}   # organic resolution


async def _mock_writer(self, state: ResearchState) -> dict:
    sections = state.get("revised_draft") or state.get("draft", [])
    return {"final_report": f"[MOCK REPORT] sections={len(sections)}"}


# ── Test ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Each test run gets its own thread so the checkpointer doesn't mix state.
    thread_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial = ResearchState(
        session_id=str(uuid.uuid4()),
        query="How does AI impact drug discovery?",
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
        # auto_approve_plan is intentionally ABSENT so the checkpoint fires
    )

    print("\n" + "=" * 68)
    print("VERIFICATION 4 — Human-in-the-Loop Plan Checkpoint")
    print("=" * 68)

    with (
        patch.object(PlannerAgent,    "run", new=_mock_planner),
        patch.object(SearcherAgent,   "run", new=_mock_searcher),
        patch.object(ReaderAgent,     "run", new=_mock_reader),
        patch.object(SynthesizerAgent,"run", new=_mock_synthesizer),
        patch.object(CriticAgent,     "run", new=_mock_critic),
        patch.object(WriterAgent,     "run", new=_mock_writer),
    ):
        # ── Part A: first invoke should pause at the checkpoint ───────────────
        print("\n--- PART A: First ainvoke() — expect interrupt at plan_approval ---")
        result1 = await app.ainvoke(initial, config=thread_cfg)

        print(f"\nFirst invoke returned.")
        print(f"  Keys with data : {[k for k, v in result1.items() if v]}")
        print(f"  __interrupt__  : {'present' if '__interrupt__' in result1 else 'ABSENT'}")

        interrupt_data = result1.get("__interrupt__", ())
        if interrupt_data:
            iv = interrupt_data[0].value if hasattr(interrupt_data[0], "value") else interrupt_data[0]
            print(f"  interrupt value checkpoint : {iv.get('checkpoint', '?')!r}")
            print(f"  interrupt value sub_questions ({len(iv.get('sub_questions', []))}):")
            for i, sq in enumerate(iv.get("sub_questions", []), 1):
                print(f"    {i}. {sq}")
        print(f"\n  draft sections after pause : {len(result1.get('draft', []))}")
        print(f"  final_report after pause   : {bool(result1.get('final_report'))}")

        # ── Part B: resume with SQ2 removed ──────────────────────────────────
        print("\n--- PART B: Resume with edited list (SQ2 removed) ---")
        approved_list = [SQ1, SQ3]   # human removed SQ2
        print(f"  Resuming with: {approved_list}")

        result2 = await app.ainvoke(Command(resume=approved_list), config=thread_cfg)

        print(f"\nSecond invoke (resume) returned.")
        print(f"  sub_questions in result : {result2.get('sub_questions', [])}")
        print(f"  draft sections          : {len(result2.get('draft', []))}")
        print(f"  draft content           :")
        for i, s in enumerate(result2.get("draft", [])):
            print(f"    [{i}] {s}")
        print(f"  final_report exists     : {bool(result2.get('final_report'))}")

        routing = result2.get("routing_history", [])
        print(f"\n  routing_history ({len(routing)} entries):")
        for e in routing:
            print(f"    {e}")

    # ── Assertions ────────────────────────────────────────────────────────────
    print("\n" + "-" * 68)
    print("Assertions:")
    failures: list[str] = []

    # A1. First invoke returned __interrupt__
    if "__interrupt__" in result1:
        print("  PASS ✓  [A1] __interrupt__ present after first ainvoke()")
    else:
        msg = "__interrupt__ key missing from first invoke result"
        print(f"  FAIL ✗  [A1] {msg}")
        failures.append(msg)

    # A2. Interrupt surfaces the three planned sub_questions
    iv = {}
    if interrupt_data:
        iv = interrupt_data[0].value if hasattr(interrupt_data[0], "value") else interrupt_data[0]
    iqs = iv.get("sub_questions", [])
    if iqs == [SQ1, SQ2, SQ3]:
        print(f"  PASS ✓  [A2] interrupt surfaces all 3 planned sub_questions")
    else:
        msg = f"interrupt sub_questions={iqs!r}, expected [SQ1, SQ2, SQ3]"
        print(f"  FAIL ✗  [A2] {msg}")
        failures.append(msg)

    # A3. Pipeline has NOT yet proceeded to fan-out (no draft)
    if not result1.get("draft"):
        print("  PASS ✓  [A3] no draft sections after pause (fan-out not started)")
    else:
        msg = f"draft={result1.get('draft')!r} — fan-out ran before checkpoint!"
        print(f"  FAIL ✗  [A3] {msg}")
        failures.append(msg)

    # B4. After resume, sub_questions == edited list
    final_sqs = result2.get("sub_questions", [])
    if final_sqs == [SQ1, SQ3]:
        print(f"  PASS ✓  [B4] sub_questions == [SQ1, SQ3] (SQ2 dropped by human)")
    else:
        msg = f"sub_questions={final_sqs!r}, expected [SQ1, SQ3]"
        print(f"  FAIL ✗  [B4] {msg}")
        failures.append(msg)

    # B5. Exactly 2 draft sections (SQ1 and SQ3 only)
    n_draft = len(result2.get("draft", []))
    if n_draft == 2:
        print(f"  PASS ✓  [B5] draft has exactly 2 sections (SQ1 and SQ3)")
    else:
        msg = f"draft has {n_draft} sections, expected 2"
        print(f"  FAIL ✗  [B5] {msg}")
        failures.append(msg)

    # B6. SQ2 does NOT appear in any draft section
    draft_text = " ".join(result2.get("draft", []))
    if SQ2 not in draft_text:
        print(f"  PASS ✓  [B6] SQ2 absent from draft (correctly dropped)")
    else:
        msg = f"SQ2 appears in draft — was not dropped by human review"
        print(f"  FAIL ✗  [B6] {msg}")
        failures.append(msg)

    # B7. final_report exists (full pipeline completed after resume)
    if result2.get("final_report"):
        print(f"  PASS ✓  [B7] final_report exists (pipeline completed after resume)")
    else:
        msg = "final_report is empty after resume"
        print(f"  FAIL ✗  [B7] {msg}")
        failures.append(msg)

    print("\n" + "=" * 68)
    if not failures:
        print("ALL ASSERTIONS PASSED ✓")
        print()
        print("Verified:")
        print("  - Graph pauses at plan_approval checkpoint after Planner")
        print("  - __interrupt__ surfaces the planned sub_questions to the caller")
        print("  - Resume with edited list propagates to fan-out")
        print("  - Only the approved sub_questions are researched")
        print("  - Dropped sub_questions (SQ2) never reach the Searcher/Synthesizer")
    else:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  • {f}")
    print("=" * 68 + "\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
