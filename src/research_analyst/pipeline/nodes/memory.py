"""Final node: persist a compact record of the report to Redis (best-effort cross-run memory)."""
from __future__ import annotations

import asyncio

from research_analyst.models import ResearchState
from research_analyst.observability.logger import get_logger
from research_analyst.observability.usage import tracker

_memory_log = get_logger("memory")


async def memory_node(state: ResearchState) -> dict:
    tracker.pop(state.get("session_id", ""))   # usage was already copied into state by the Writer
    return await memory_node_inner(state)


async def memory_node_inner(state: ResearchState) -> dict:
    """
    Persist a compact record of the completed report to Redis for cross-run
    memory.  Runs as the final graph node (after Writer).

    Best-effort: a Redis failure is logged as a WARNING but does NOT crash the
    pipeline.  The final_report is still fully available in state.

    Stored fields per report: report_id, session_id, query, sub_questions,
    executive_summary (first 500 chars), timestamp, timestamp_human.
    """
    try:
        from research_analyst.memory.redis_store import save_report_record
        report_id = await asyncio.to_thread(
            save_report_record,
            session_id=state.get("session_id", "unknown"),
            query=state.get("query", ""),
            final_report=state.get("final_report", ""),
            sub_questions=list(state.get("sub_questions", [])),
        )
        _memory_log.info(
            f"Report persisted to Redis: report_id={report_id[:8]}",
            step="memory_write",
            session_id=state.get("session_id", ""),
        )
        return {"routing_history": [f"memory | report saved | id={report_id[:8]}"]}
    except Exception as exc:
        _memory_log.warning(
            f"Memory write failed (Redis unavailable?): {exc}",
            step="memory_write_fail",
            session_id=state.get("session_id", ""),
        )
        return {}
