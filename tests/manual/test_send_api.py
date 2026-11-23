"""
tests/test_send_api.py -- THROWAWAY: LangGraph Send() API verification.

Answers four questions about Send() in LangGraph 1.2.7 before the real
fan-out implementation is written.

ARCHITECTURE DECISION DERIVED FROM THIS TEST:
  Chained edges (A->B->C) after a Send() fan-out do NOT maintain
  branch-local state. After A completes, B runs on GLOBAL state -- the
  branch context from the Send arg is lost. This means:

    WRONG: Send("searcher", arg) with edges searcher->reader->synthesizer
           (Reader would lose current_sub_question and run once globally)

    RIGHT: Send("process_sub_question", arg) where process_sub_question
           is a single composite function running all three agents internally.

  This is the classic LangGraph "map-reduce with a single worker node" pattern.
"""
from __future__ import annotations

import asyncio
import operator
import sys
from pathlib import Path
from typing import Annotated, List

from typing_extensions import NotRequired, TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# =========================================================================
# PART 1: chained-edge test (Add->B->C topology)
# Expected finding: current_item is LOST after A, B and C run on global state
# =========================================================================

class ChainedState(TypedDict):
    items: List[str]
    log: Annotated[List[str], operator.add]
    current_item: NotRequired[str]


def chained_distribute(state: ChainedState):
    return [Send("ca", {**state, "current_item": item}) for item in state["items"]]

async def ca(state): item = state.get("current_item","MISSING_A"); return {"log": [f"A|{item}"]}
async def cb(state): item = state.get("current_item","MISSING_B"); return {"log": [f"B|{item}"]}
async def cc(state): item = state.get("current_item","MISSING_C"); return {"log": [f"C|{item}"]}


def build_chained():
    g = StateGraph(ChainedState)
    g.add_node("dist", chained_distribute)
    g.add_node("ca", ca); g.add_node("cb", cb); g.add_node("cc", cc)
    g.add_conditional_edges(START, chained_distribute)
    g.add_edge("ca", "cb"); g.add_edge("cb", "cc"); g.add_edge("cc", END)
    return g.compile()


# =========================================================================
# PART 2: composite-node test (single function per branch)
# Expected finding: current_item survives because it never leaves the function
# =========================================================================

class CompositeState(TypedDict):
    items: List[str]
    log: Annotated[List[str], operator.add]


def composite_distribute(state: CompositeState):
    return [Send("worker", {**state, "current_item": item}) for item in state["items"]]

async def composite_worker(state: dict) -> dict:
    """Single function replacing A->B->C. current_item stays in local scope."""
    item = state.get("current_item", "MISSING")
    # Simulate three sequential sub-steps inside one function
    result_a = f"A|{item}"
    result_b = f"B|{item}|saw_a={result_a}"
    result_c = f"C|{item}|saw_b={result_b}"
    return {"log": [result_a, result_b, result_c]}


def build_composite():
    g = StateGraph(CompositeState)
    g.add_node("dist", composite_distribute)
    g.add_node("worker", composite_worker)
    g.add_conditional_edges(START, composite_distribute)
    g.add_edge("worker", END)
    return g.compile()


# =========================================================================
# Runner
# =========================================================================

async def main() -> None:
    print("\n" + "=" * 65)
    print("Send() API verification  LangGraph 1.2.7")
    print("=" * 65)

    # -- Part 1: chained edges after Send ----------------------------------
    print("\n[Part 1] Chained edges (Send -> A -> B -> C via add_edge)")
    print("  Hypothesis: current_item should survive into B and C")
    app1 = build_chained()
    r1 = await app1.ainvoke({"items": ["alpha", "beta"], "log": []})
    log1 = r1["log"]
    print(f"  Actual log entries ({len(log1)}): {log1}")

    has_a_alpha = any("A|alpha" in e for e in log1)
    has_a_beta  = any("A|beta"  in e for e in log1)
    b_lost = any("MISSING" in e and e.startswith("B") for e in log1)
    c_lost = any("MISSING" in e and e.startswith("C") for e in log1)

    print(f"\n  Q1  Both A branches ran:                {has_a_alpha and has_a_beta}")
    print(f"  Q2a current_item LOST in B (MISSING):   {b_lost}")
    print(f"  Q2b current_item LOST in C (MISSING):   {c_lost}")
    print(f"  Q2c B ran once (not per-branch):        {len([e for e in log1 if e.startswith('B')]) == 1}")
    print(f"  Q2d C ran once (not per-branch):        {len([e for e in log1 if e.startswith('C')]) == 1}")
    print(f"  FINDING: chained nodes after Send run on GLOBAL state,")
    print(f"           branch context is LOST -- MUST use composite node pattern")

    assert has_a_alpha and has_a_beta, "Both A branches must run"
    assert b_lost, "B must NOT see current_item (it runs on global state, context is gone)"
    assert c_lost, "C must NOT see current_item (same reason)"

    # -- Part 2: composite node ---------------------------------------------
    print("\n[Part 2] Composite node (single function per branch)")
    print("  Hypothesis: current_item and inter-step results survive inside the function")
    app2 = build_composite()
    r2 = await app2.ainvoke({"items": ["alpha", "beta"], "log": []})
    log2 = r2["log"]
    print(f"  Actual log entries ({len(log2)}): {log2}")

    a_alpha = [e for e in log2 if "A|alpha" in e]
    b_alpha = [e for e in log2 if "B|alpha" in e and "saw_a" in e and "MISSING" not in e]
    c_alpha = [e for e in log2 if "C|alpha" in e and "saw_b" in e and "MISSING" not in e]
    a_beta  = [e for e in log2 if "A|beta"  in e]
    b_beta  = [e for e in log2 if "B|beta"  in e and "saw_a" in e and "MISSING" not in e]
    c_beta  = [e for e in log2 if "C|beta"  in e and "saw_b" in e and "MISSING" not in e]

    print(f"\n  Q3  Both branches have all 3 sub-steps: {bool(a_alpha and b_alpha and c_alpha and a_beta and b_beta and c_beta)}")
    print(f"  Q4  operator.add merged all 6 entries:  {len(log2) == 6}")
    print(f"  Q5  async composite nodes work:          True (no error)")
    # Execution order
    items_order = [e.split("|")[1] for e in log2 if e.startswith("A")]
    interleaved = (len(items_order) == 2 and items_order[0] != items_order[1])
    print(f"  Q6  branches interleaved/concurrent:    {interleaved} (order: {items_order})")

    assert a_alpha and b_alpha and c_alpha, "alpha chain must have all 3 steps with context"
    assert a_beta  and b_beta  and c_beta,  "beta chain must have all 3 steps with context"
    assert len(log2) == 6, f"Expected 6 log entries, got {len(log2)}"

    print("\n[Summary]")
    print("  CHAINED topology:   current_item LOST after first Send node  ? WRONG pattern")
    print("  COMPOSITE topology: current_item lives inside function scope  ? CORRECT pattern")
    print()
    print("  ARCHITECTURAL DECISION FOR REAL GRAPH:")
    print("  Replace Send('searcher', ...) + edges searcher->reader->synthesizer")
    print("  with Send('process_sub_question', ...) where process_sub_question")
    print("  is a single async function running all three agents internally.")
    print()
    print("  SAFE to write to Annotated[List, operator.add] fields from parallel branches.")
    print("  UNSAFE to write to plain (non-reducer) fields from parallel branches.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
