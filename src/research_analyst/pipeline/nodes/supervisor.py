"""Fan-in observer: records a summary and fails clearly if every branch failed."""
from __future__ import annotations

from research_analyst.models import ResearchState


def supervisor_node(state: ResearchState) -> dict:
    """
    Fan-in observer -- runs after all parallel branches complete.

    Records a summary entry in routing_history, then the graph proceeds
    unconditionally to the Writer via a plain add_edge().

    There is no graph-level retry path.  Zero-results retries are handled
    inside process_sub_question (branch-local, capped at 1 attempt).
    A supervisor->searcher conditional edge was removed (deliberate
    architectural change): current_sub_question is set only in Send() args
    (branch-local) and never written back to global state, so any go_searcher
    condition would be permanently False.  See module docstring and
    tests/test_v2_retry_parallel.py.
    """
    n_sq    = len(state.get("sub_questions", []))
    n_facts = len(state.get("extracted_facts", []))
    n_gaps  = len(state.get("unresolved_gaps", []))

    if not state.get("draft"):
        raise RuntimeError(
            f"All {n_sq} research branch(es) failed — no draft sections were produced. "
            "See branch_failed entries in logs/agent_traces.jsonl."
        )

    log_entry = (
        f"supervisor | {n_sq} branch(es) complete -> debate"
        f" | facts={n_facts} gaps={n_gaps}"
    )
    return {"routing_history": [log_entry]}
