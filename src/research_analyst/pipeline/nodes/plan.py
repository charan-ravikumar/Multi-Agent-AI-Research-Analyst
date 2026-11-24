"""Plan approval: cross-run memory lookup + the human checkpoint (interrupt/resume)."""
from __future__ import annotations

import asyncio

from langgraph.types import interrupt

from research_analyst.models import ResearchState
from research_analyst.observability.logger import get_logger

_plan_log = get_logger("plan")


async def plan_approval_node(state: ResearchState) -> dict:
    """
    Human checkpoint between Planner output and fan-out.

    Also performs a best-effort cross-run memory lookup (keyword matching,
    NOT semantic search — see memory/redis_store.find_related_reports) and
    populates state["related_past_reports"] with any topically overlapping
    past reports found in Redis.

    Behaviour:
      - auto_approve_plan=True  (automated tests / CI): returns immediately,
        sub_questions unchanged.  related_past_reports still populated.
      - auto_approve_plan absent or False: calls interrupt() to surface the
        sub_questions (and any related past work) to the caller.  Execution
        pauses until the graph is resumed via Command(resume=<approved_list>).
        The resume value replaces state["sub_questions"] so the fan-out
        dispatches the approved set.

    Verified mechanism (tests/_verify_interrupt.py):
      interrupt(value) pauses the graph and returns the value in
      result["__interrupt__"].  Command(resume=v) resumes; interrupt()
      returns v inside the node.  The node re-runs from the top on resume
      but interrupt() returns immediately with the stored value.
      Requires graph compiled with a checkpointer AND thread_id in config.
    """
    sub_questions: list[str] = list(state.get("sub_questions", []))

    # ── cross-run memory lookup ──────────────────────────────────────────────
    # Best-effort: if Redis is unavailable the pipeline continues without
    # related-report context (graceful degradation).
    related: list[dict] = []
    try:
        from research_analyst.memory.redis_store import find_related_reports
        related = await asyncio.to_thread(
            find_related_reports, state.get("query", "")
        )
        if related:
            _plan_log.info(
                f"Found {len(related)} related past report(s) for query={state.get('query', '')[:60]!r}",
                step="memory_lookup",
                session_id=state.get("session_id", ""),
            )
    except Exception as exc:
        _plan_log.warning(
            f"Related-report lookup failed (Redis unavailable?): {exc}",
            step="memory_lookup_fail",
            session_id=state.get("session_id", ""),
        )

    if state.get("auto_approve_plan", False):
        # Automated / test path: skip human review.
        return {
            "related_past_reports": related,
            "routing_history": [
                f"plan | auto_approved"
                f" | sub_questions={len(sub_questions)}"
                f" | related_found={len(related)}"
            ],
        }

    # Human-in-the-loop path: surface plan + related past work for review.
    # NOTE: Because LangGraph re-runs the node from the top on resume,
    # this block executes twice: once before the pause and once on resume
    # (where interrupt() immediately returns the stored value).
    print("\n" + "=" * 62)
    print("HUMAN CHECKPOINT \u2014 Research Plan Approval")
    print("=" * 62)
    print(f"Query: {state.get('query', '')}")
    print(f"\nProposed sub-questions ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    if related:
        print(f"\nRelated past reports ({len(related)}):")
        for r in related[:3]:
            print(f"  [{r.get('timestamp_human','?')}] {r.get('query','')[:80]}")
            print(f"    Shared tokens: {r.get('_overlap_tokens', [])}")
    print("\nResume with: Command(resume=<approved_list>)")
    print("  Approve as-is : Command(resume=state['sub_questions'])")
    print("  Edit          : Command(resume=[sq1, sq2, ...])")
    print("=" * 62 + "\n")

    approved: list[str] = interrupt({
        "checkpoint": "plan_approval",
        "query": state.get("query", ""),
        "sub_questions": sub_questions,
        "related_past_reports": related,
        "message": (
            "Review the proposed sub-questions. "
            "Resume with the approved list (edit as needed)."
        ),
    })

    return {
        "sub_questions": approved,
        "related_past_reports": related,
        "routing_history": [
            f"plan | human_approved"
            f" | approved={len(approved)}/{len(sub_questions)}"
            f" | related_found={len(related)}"
        ],
    }
