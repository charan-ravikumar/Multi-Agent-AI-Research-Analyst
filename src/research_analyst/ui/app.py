"""Top-level Streamlit app: page config, tabs, phase dispatch."""
from __future__ import annotations

import streamlit as st

from research_analyst.ui import queued
from research_analyst.ui import state as S
from research_analyst.ui import theme
from research_analyst.ui.monitor import render_monitor
from research_analyst.ui.about import ABOUT_MD
from research_analyst.ui.gallery import render_gallery
from research_analyst.ui.views import approval, cancelled, done, error, idle, planning, researching

PHASE_DISPATCH = {
    "idle":              idle.phase_idle,
    "planning":          planning.phase_planning,
    "awaiting_approval": approval.phase_awaiting_approval,
    "researching":       researching.phase_researching,
    "done":              done.phase_done,
    "error":             error.phase_error,
    "cancelled":         cancelled.phase_cancelled,
}


def main() -> None:
    st.set_page_config(
        page_title="AI Research Analyst",
        layout="wide",
    )
    theme.apply()

    if "phase" not in st.session_state:
        st.session_state.phase = "idle"

    dispatch = dict(PHASE_DISPATCH)
    if queued.enabled():                       # a worker runs the pipeline; these two phases only watch it
        dispatch.update(planning=queued.phase_planning, researching=queued.phase_researching)
        queued.reattach()

    tab_research, tab_gallery, tab_monitor, tab_about = st.tabs(["Research", "Sample reports", "Monitor", "About"])
    with tab_about:
        st.markdown(ABOUT_MD)
    with tab_gallery:
        render_gallery()
    with tab_monitor:
        render_monitor()
    with tab_research:
        handler = dispatch.get(st.session_state.phase)
        if handler is None:
            st.error(f"Unknown phase: {st.session_state.phase!r}")
            S.reset_state()
            st.rerun()
            return
        handler()
