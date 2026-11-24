"""Top-level Streamlit app: page config, tabs, phase dispatch."""
from __future__ import annotations

import streamlit as st

from research_analyst.ui import state as S
from research_analyst.ui.about import ABOUT_MD
from research_analyst.ui.views import approval, done, error, idle, planning, researching

PHASE_DISPATCH = {
    "idle":              idle.phase_idle,
    "planning":          planning.phase_planning,
    "awaiting_approval": approval.phase_awaiting_approval,
    "researching":       researching.phase_researching,
    "done":              done.phase_done,
    "error":             error.phase_error,
}


def main() -> None:
    st.set_page_config(
        page_title="AI Research Analyst",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 AI Research Analyst")
    st.caption("Ask a question — get a sourced report. Sub-questions, search, reading and writing run in parallel.")

    if "phase" not in st.session_state:
        st.session_state.phase = "idle"

    tab_research, tab_about = st.tabs(["🔬 Research", "ℹ️ About this app"])
    with tab_about:
        st.markdown(ABOUT_MD)
    with tab_research:
        handler = PHASE_DISPATCH.get(st.session_state.phase)
        if handler is None:
            st.error(f"Unknown phase: {st.session_state.phase!r}")
            S.reset_state()
            st.rerun()
            return
        handler()
