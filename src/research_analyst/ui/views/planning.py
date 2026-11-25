"""Planning phase: blocking run up to the plan-approval interrupt."""
from __future__ import annotations

from typing import Any

import asyncio

import streamlit as st

from research_analyst.ui import state as S


def phase_planning() -> None:
    """
    Blocking invocation of the graph until it hits the plan-approval interrupt.
    asyncio.run() is safe here: Streamlit does not hold a running event loop
    between script reruns (verified against 1.58.0).
    """
    from research_analyst.models.research_state import new_research_state  # noqa: PLC0415

    st.markdown(f"**Query:** {st.session_state.query}")

    with st.spinner("Generating research plan — this takes around 20–60 seconds…"):
        initial = new_research_state(
            st.session_state.session_id, st.session_state.query,
            review_rounds=st.session_state.get("review_rounds", 0),
        )
        cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            from research_analyst.observability import tracing  # noqa: PLC0415

            async def _plan() -> Any:
                with tracing.run_span("planning", st.session_state.session_id, st.session_state.query):
                    return await S.get_graph().ainvoke(initial, config=cfg)

            result = asyncio.run(_plan())
        except Exception as exc:
            st.session_state.phase = "error"
            st.session_state.error = str(exc)
            st.rerun()
            return

    interrupted = result.get("__interrupt__")
    if not interrupted:
        # Graph ran to completion without pausing (should not happen with
        # auto_approve_plan=False, but handle it gracefully).
        st.session_state.phase       = "done"
        st.session_state.final_state = result
        st.rerun()
        return

    st.session_state.plan_result = result
    st.session_state.phase       = "awaiting_approval"
    st.rerun()
