"""Session-state helpers and constants shared by all views."""
from __future__ import annotations

from typing import Any

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
LOG_FILE = ROOT / "logs" / "agent_traces.jsonl"
POLL_INTERVAL = 3   # seconds between progress refreshes during the research phase
TRACE_TAIL = 10     # number of recent trace events to show


@st.cache_resource
def load_graph() -> Any:
    """Import and cache the compiled LangGraph app across all Streamlit reruns."""
    from research_analyst.pipeline.graph import app  # noqa: PLC0415
    return app


def get_graph() -> Any:
    return load_graph()


SESSION_KEYS = (
    "phase", "query", "session_id", "thread_id",
    "plan_result", "final_state",
    "bg_thread", "bg_result",
    "error",
    "sq_textarea", "_sq_default",
    "query_input", "review_loop", "review_rounds",
)


def reset_state() -> None:
    """Clear all per-run session state.  The cached graph (@st.cache_resource)
    is intentionally preserved — no need to re-import on each run."""
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def cancel_run() -> None:
    """Button callback: stop the current research run."""
    from research_analyst.observability.progress import progress  # noqa: PLC0415
    from research_analyst.ui.runner import request_cancel  # noqa: PLC0415
    if request_cancel(st.session_state.get("bg_result")):
        progress.cancel(st.session_state.get("session_id", ""))
