"""
tests/test_v1_send_isolation.py  —  Verification 1

Throwaway graph that answers two questions about LangGraph Send():

  (a) STATE ISOLATION
      Does {**state, "branch_id": X} in _fan_out give each branch its own
      independent copy, or do branches share references to nested objects?

      Mechanism tested: we put a mutable list in state and each branch
      appends to it IN-PLACE.  The "snapshot_before" recorded on entry to
      each branch reveals whether earlier branches' mutations are visible.

      - If LangGraph deep-copies nested objects → every snapshot_before is []
      - If LangGraph shallow-copies (same reference) → later branches see
        earlier mutations in their snapshot_before

      Either result is reported honestly; the pipeline is safe regardless
      because _process_sub_question never mutates shared objects in-place.

  (b) OPERATOR.ADD CORRECTNESS
      Does operator.add accumulate ALL N branches' contributions without
      dropping any, regardless of completion order?

      Each branch returns one item in an Annotated[List[str], operator.add]
      field.  After fan-in we verify all 3 items are present (no drops,
      no duplicates).

Run with:
    .venv/Scripts/python.exe tests/test_v1_send_isolation.py
"""
from __future__ import annotations

import asyncio
import operator
import sys
from pathlib import Path
from typing import Annotated, List

from typing_extensions import TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# ── State ─────────────────────────────────────────────────────────────────────

class IsolationState(TypedDict):
    branch_id: str          # set per-branch by the Send arg
    nested_list: list       # mutable nested object — the isolation test target
    results: Annotated[List[str], operator.add]       # operator.add accumulator
    mutation_log: Annotated[List[str], operator.add]  # per-branch mutation evidence


# ── Graph nodes ───────────────────────────────────────────────────────────────

def fan_out(state: IsolationState) -> list:
    """Fan out over 3 items.  Uses the exact same {**state, key: val} pattern
    as the real _fan_out in orchestrator/graph.py."""
    return [
        Send("branch", {**state, "branch_id": item})
        for item in ["alpha", "beta", "gamma"]
    ]


async def branch_node(state: IsolationState) -> dict:
    """
    (a) Mutation isolation test:
        - Records what nested_list looks like ON ENTRY  (snapshot_before)
        - Appends its own marker to nested_list IN-PLACE
        - Records what it looks like AFTER our own write (snapshot_after)

    If nested_list is a shared reference, a branch that starts AFTER
    another branch has already mutated it will see a non-empty snapshot_before.

    (b) operator.add test:
        Returns one item in 'results'; all 3 must appear in the final state.
    """
    bid = state["branch_id"]
    nl = state["nested_list"]           # may be the same object across branches

    snapshot_before = list(nl)          # copy of what we see on arrival
    nl.append(f"written_by_{bid}")      # in-place mutation
    snapshot_after = list(nl)           # what we see after our own write

    return {
        "results": [f"result_{bid}"],
        "mutation_log": [
            f"{bid}:  before={snapshot_before}  after={snapshot_after}"
        ],
    }


# ── Build and compile ─────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(IsolationState)
    graph.add_node("branch", branch_node)
    graph.add_conditional_edges(START, fan_out)
    graph.add_edge("branch", END)
    return graph.compile()


_app = build_graph()


# ── Runner ────────────────────────────────────────────────────────────────────

async def main() -> None:
    initial = IsolationState(
        branch_id="",
        nested_list=[],      # empty mutable list; all Send args point at this object
        results=[],
        mutation_log=[],
    )

    result = await _app.ainvoke(initial)

    print("\n" + "=" * 68)
    print("VERIFICATION 1 — Send() State Isolation + operator.add")
    print("=" * 68)

    # ── (b) operator.add ─────────────────────────────────────────────────────
    print("\n(b) operator.add — all 3 branch contributions must be in 'results':")
    for r in sorted(result["results"]):
        print(f"     {r}")

    expected = {"result_alpha", "result_beta", "result_gamma"}
    actual   = set(result["results"])
    add_ok   = actual == expected
    print(f"\n     Expected : {sorted(expected)}")
    print(f"     Got      : {sorted(actual)}")
    if add_ok:
        print("     PASS ✓  — operator.add merged all 3 contributions, none dropped")
    else:
        missed = expected - actual
        extra  = actual - expected
        print(f"     FAIL ✗  — missed={missed}  extra={extra}")

    # ── (a) State isolation ───────────────────────────────────────────────────
    print("\n(a) State isolation — snapshot_before shows what each branch sees on entry:")
    for entry in sorted(result["mutation_log"]):
        print(f"     {entry}")

    # Parse snapshot_before for each branch
    entries_with_non_empty_before = [
        e for e in result["mutation_log"]
        if "before=[]" not in e
    ]

    print()
    if not entries_with_non_empty_before:
        print("     Result : ISOLATED ✓")
        print("              Every branch entered with nested_list=[].")
        print("              LangGraph gave each branch its own copy of the nested object.")
        print("              (or branches ran sequentially — see elapsed times above)")
    else:
        print("     Result : SHARED REFERENCE ⚠")
        print("              Some branches saw mutations from earlier branches on entry.")
        print("              This means {**state, ...} is a SHALLOW copy —")
        print("              nested mutable objects are shared across branches.")
        print()
        print("              For the research pipeline this is SAFE because")
        print("              _process_sub_question never mutates lists in-place;")
        print("              it always creates new lists with the + operator.")
        print("              But be aware: any future branch code that mutates")
        print("              state['search_results'].append(...) would race.")

    print("\n" + "=" * 68)
    if add_ok:
        print("VERDICT (b): PASS — operator.add is correct")
    else:
        print("VERDICT (b): FAIL — operator.add is broken — investigate immediately")
    print("=" * 68 + "\n")

    # Hard-fail only on operator.add (that's the correctness guarantee)
    assert add_ok, f"operator.add dropped contributions: expected {expected}, got {actual}"


if __name__ == "__main__":
    asyncio.run(main())
