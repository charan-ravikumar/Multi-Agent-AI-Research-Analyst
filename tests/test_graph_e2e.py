"""
tests/test_graph_e2e.py

End-to-end test of the compiled LangGraph pipeline.
Invokes the graph via app.ainvoke() — NOT by calling agents directly.
Run from the project root:

    .venv/Scripts/python.exe tests/test_graph_e2e.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.graph import app
from models import ResearchState


def make_initial_state(query: str) -> ResearchState:
    """
    Minimal valid ResearchState to feed into the graph.
    Only query and session_id are meaningful — everything else is empty.
    The Planner will populate sub_questions; downstream agents fill the rest.
    """
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
        routing_history=[],
        auto_approve_plan=True,  # skip human checkpoint in automated e2e test
    )


async def main() -> None:
    query = "What is the impact of AI on drug discovery timelines?"

    print(f"\n{'='*65}")
    print(f"Query   : {query}")
    print(f"Graph   : {[n for n in app.get_graph().nodes]}")
    print(f"{'='*65}\n")

    initial = make_initial_state(query)
    print("Invoking graph via app.ainvoke()...\n")

    _thread_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result: ResearchState = await app.ainvoke(initial, config=_thread_cfg)

    # ── report the pipeline outputs ───────────────────────────────────────────
    print(f"{'─'*65}")
    n_sq = len(result.get('sub_questions', []))
    print(f"Sub-questions planned  : {n_sq}")
    print(f"Sub-questions in draft : {len(result.get('draft', []))}")
    print(f"Search results         : {len(result.get('search_results', []))}")
    print(f"Extracted facts        : {len(result.get('extracted_facts', []))}")
    print(f"Citations              : {len(result.get('citations', []))}")
    print(f"Contradictions         : {len(result.get('contradictions', []))}")
    print(f"Unresolved gaps        : {len(result.get('unresolved_gaps', []))}")
    print(f"{'─'*65}")
    print("Routing history:")
    for entry in result.get('routing_history', []):
        print(f"  {entry}")
    print(f"{'─'*65}\n")

    print("FINAL REPORT:")
    print(f"{'='*65}\n")
    print(result.get("final_report", "(empty)"))
    print(f"\n{'='*65}\n")

    # ── assertions ────────────────────────────────────────────────────────────
    assert result.get("sub_questions"), "Planner must produce sub_questions"
    assert result.get("search_results"), "Searcher must populate search_results"
    assert result.get("extracted_facts"), "Reader must populate extracted_facts"
    assert result.get("draft"), "Synthesizer must produce a draft"
    assert result.get("citations"), "Synthesizer must produce citations"

    # All sub-questions should have produced a draft section
    assert len(result.get("draft", [])) == n_sq, (
        f"Expected {n_sq} draft sections (one per sub-question), "
        f"got {len(result.get('draft', []))}"
    )

    report = result.get("final_report", "")
    assert report.strip(), "Writer must produce a non-empty final_report"

    report_lower = report.lower()
    for section in ["executive summary", "key findings", "detailed analysis",
                    "contradictions", "knowledge gaps", "references"]:
        assert section in report_lower, \
            f"final_report missing required section: '{section}'"

    print(f"All assertions passed.")
    print(f"Pipeline: {n_sq} sub-questions processed in parallel via fan-out.")


if __name__ == "__main__":
    asyncio.run(main())
