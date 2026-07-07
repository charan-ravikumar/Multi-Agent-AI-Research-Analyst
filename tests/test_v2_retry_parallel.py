"""
tests/test_v2_retry_parallel.py  —  Verification 2

Tests that the per-branch zero-results retry works correctly when
multiple branches are fanning into the supervisor concurrently.

The entire LLM + search stack is mocked so this runs fully offline.

SETUP
-----
Three sub-questions are dispatched in parallel via Send():
  SQ1  "How does ML accelerate target identification?"
  SQ2  "What deep learning methods are used in drug design?"
  SQ3  "How are AI safety risks managed in clinical trials?"  ← ZERO RESULTS

The mock searcher:
  • SQ1, SQ2 → returns one SearchResult each (normal)
  • SQ3       → returns empty list (first call AND retry call)
               → routing_history must show exactly one retry entry for SQ3

WHAT WE VERIFY
--------------
After app.ainvoke():

  1. routing_history contains exactly ONE retry_searcher entry, for SQ3 only.
  2. No retry entries exist for SQ1 or SQ2.
  3. routing_history contains a supervisor fan-in entry that routes -> debate.
  4. draft has exactly 3 sections (one per branch; SQ3 gets an honest empty-data synthesis).
  5. SQ3 was searched exactly twice (initial + 1 branch-local retry).
  6. SQ1 and SQ2 were each searched exactly once (no spurious retries).

WHY THIS MATTERS
----------------
The composite node _process_sub_question handles zero-results retry internally.
The Supervisor sees the accumulated state AFTER all branches have merged — it
must not see SQ3's current_sub_question or accidentally retry at graph level.

Run with:
    .venv/Scripts/python.exe tests/test_v2_retry_parallel.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.planner import PlannerAgent
from agents.critic import CriticAgent
from agents.reader import ReaderAgent
from agents.searcher import SearcherAgent
from agents.synthesizer import SynthesizerAgent
from agents.writer import WriterAgent
from models import ResearchState
from models.fact import Fact
from models.research_plan import ResearchPlan
from models.search_result import SearchResult
from orchestrator.graph import app

# ── Fixed sub-questions (planner is mocked to return these exactly) ───────────
SQ1 = "How does ML accelerate target identification?"
SQ2 = "What deep learning methods are used in drug design?"
SQ3 = "How are AI safety risks managed in clinical trials?"   # ← zero results


# ── Mock implementations ──────────────────────────────────────────────────────

async def _mock_planner(self, state: ResearchState) -> dict:
    """Returns a fixed 3-sub-question plan without calling the LLM."""
    plan = ResearchPlan(
        query=state["query"],
        sub_questions=[SQ1, SQ2, SQ3],
    )
    return {**state, "research_plan": plan, "sub_questions": [SQ1, SQ2, SQ3]}


# Track how many times the searcher is called per sub-question
_searcher_call_counts: dict[str, int] = {}


async def _mock_searcher(self, state: ResearchState) -> dict:
    """
    SQ1, SQ2 → one result each.
    SQ3       → empty on EVERY call (first pass AND the internal retry).
    """
    sq = state.get("current_sub_question", "")
    _searcher_call_counts[sq] = _searcher_call_counts.get(sq, 0) + 1

    if sq == SQ3:
        return {
            "search_results": [],
            "failed_sub_questions": [sq],
        }

    result = SearchResult(
        sub_question=sq,
        url=f"https://example.com/{sq[:20].replace(' ', '-')}",
        title=f"Mock result for: {sq[:40]}",
        snippet=f"A mock snippet answering: {sq[:60]}",
    )
    return {
        "search_results": [result],
        "failed_sub_questions": [],
    }


async def _mock_reader(self, state: ResearchState) -> dict:
    """Returns one fact for sub-questions that had results; empty for SQ3."""
    sq = state.get("current_sub_question", "")
    results_for_sq = [
        r for r in state.get("search_results", []) if r.sub_question == sq
    ]
    if not results_for_sq:
        return {"extracted_facts": []}
    fact = Fact(
        sub_question=sq,
        content=f"Mock fact for: {sq}",
        source_url=results_for_sq[0].url,
        source_title=results_for_sq[0].title,
        confidence=0.9,
    )
    return {"extracted_facts": [fact]}


async def _mock_synthesizer(self, state: ResearchState) -> dict:
    sq = state.get("current_sub_question", "")
    facts = [f for f in state.get("extracted_facts", []) if f.sub_question == sq]
    note = "(no data — zero search results)" if not facts else f"({len(facts)} fact(s))"
    return {
        "draft": [f"[MOCK SYNTHESIS] {sq}  {note}"],
        "citations": [],
        "contradictions": [],
        "unresolved_gaps": [f"Gap: no data for {sq}"] if not facts else [],
    }


async def _mock_writer(self, state: ResearchState) -> dict:
    return {"final_report": "[MOCK REPORT] pipeline completed"}


async def _mock_critic(self, state: ResearchState) -> dict:
    """No objections — organic resolution so debate terminates immediately."""
    return {"critic_objections": []}


# ── Test ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    global _searcher_call_counts
    _searcher_call_counts = {}

    initial = ResearchState(
        session_id=str(uuid.uuid4()),
        query="What is the role of AI in drug discovery?",
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

    # ── Print full routing_history ────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("VERIFICATION 2 — Supervisor retry under parallel fan-in")
    print("=" * 68)

    routing = result.get("routing_history", [])
    print(f"\nrouting_history ({len(routing)} entries):")
    for i, entry in enumerate(routing):
        print(f"  [{i}] {entry}")

    print(f"\nSearcher call counts per sub-question:")
    for sq, count in sorted(_searcher_call_counts.items()):
        retry_label = " (initial + 1 retry)" if count == 2 else ""
        print(f"  {count}x  {sq}{retry_label}")

    print(f"\ndraft sections ({len(result.get('draft', []))}):")
    for i, section in enumerate(result.get("draft", [])):
        print(f"  [{i}] {section}")

    print(f"current_sub_question (post-fan-in): {result.get('current_sub_question', '(absent)')!r}")

    # ── Assertions ────────────────────────────────────────────────────────────
    print("\n" + "-" * 68)
    print("Assertions:")
    failures: list[str] = []

    # 1. Exactly one retry entry, for SQ3 only
    retry_entries = [e for e in routing if "retry_searcher" in e]
    sq3_retries   = [e for e in retry_entries if SQ3 in e]
    sq1_retries   = [e for e in retry_entries if SQ1 in e]
    sq2_retries   = [e for e in retry_entries if SQ2 in e]

    if len(sq3_retries) == 1:
        print(f"  PASS ✓  [1] Exactly one retry_searcher entry, for SQ3")
    else:
        msg = f"Expected 1 retry entry for SQ3, got {len(sq3_retries)}: {sq3_retries}"
        print(f"  FAIL ✗  [1] {msg}")
        failures.append(msg)

    if not sq1_retries and not sq2_retries:
        print(f"  PASS ✓  [2] No retry entries for SQ1 or SQ2")
    else:
        msg = f"SQ1 retries={sq1_retries}  SQ2 retries={sq2_retries}"
        print(f"  FAIL ✗  [2] {msg}")
        failures.append(msg)

    # 3. Supervisor routed to writer
    supervisor_entries = [e for e in routing if "supervisor" in e and "branch(es) complete" in e]
    debate_routes = [e for e in supervisor_entries if "-> debate" in e]
    writer_direct_routes = [e for e in supervisor_entries if "-> writer" in e]

    if debate_routes and not writer_direct_routes:
        print(f"  PASS \u2713  [3] Supervisor routed -> debate (as expected post-Stage-5)")
    else:
        msg = f"debate_routes={debate_routes}  writer_direct_routes={writer_direct_routes}"
        print(f"  FAIL ✗  [3] {msg}")
        failures.append(msg)

    # 4. draft has exactly 3 sections
    n_draft = len(result.get("draft", []))
    if n_draft == 3:
        print(f"  PASS ✓  [4] draft has 3 sections (one per sub-question)")
    else:
        msg = f"draft has {n_draft} sections, expected 3"
        print(f"  FAIL ✗  [4] {msg}")
        failures.append(msg)

    # 5. SQ3 was searched exactly twice (initial + 1 retry)
    sq3_calls = _searcher_call_counts.get(SQ3, 0)
    if sq3_calls == 2:
        print(f"  PASS ✓  [5] SQ3 searched 2× (1 initial + 1 retry)")
    else:
        msg = f"SQ3 searched {sq3_calls}×, expected 2"
        print(f"  FAIL ✗  [5] {msg}")
        failures.append(msg)

    # 6. SQ1, SQ2 each searched exactly once
    sq1_calls = _searcher_call_counts.get(SQ1, 0)
    sq2_calls = _searcher_call_counts.get(SQ2, 0)
    if sq1_calls == 1 and sq2_calls == 1:
        print(f"  PASS ✓  [6] SQ1 and SQ2 each searched exactly once (no spurious retries)")
    else:
        msg = f"SQ1 calls={sq1_calls}, SQ2 calls={sq2_calls}, expected 1 each"
        print(f"  FAIL ✗  [6] {msg}")
        failures.append(msg)

    print("\n" + "=" * 68)
    if not failures:
        print("ALL ASSERTIONS PASSED ✓")
        print()
        print("Conclusion: the per-branch retry is correctly scoped to the zero-result")
        print("sub-question only.  The other branches are unaffected.  The Supervisor")
        print("sees the fully-accumulated fan-in state and routes to the Writer without")
        print("any graph-level retry path (that path has been removed as dead code).")
    else:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  • {f}")
    print("=" * 68 + "\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
