"""Shown after the user stops a run."""
from __future__ import annotations

import streamlit as st

from research_analyst.observability.usage import tracker
from research_analyst.ui import state as S


def phase_cancelled() -> None:
    st.warning("**Research stopped.** No report was produced.")
    u = tracker.summary(st.session_state.get("session_id", ""))
    if u.get("calls"):
        st.caption(f"Used before stopping: {u['total_tokens']:,} tokens in {u['calls']} LLM calls "
                   "(requests already in flight may still complete and count).")
    st.markdown(f"**Query:** {st.session_state.get('query', '')}")
    if st.button("Start over", type="primary"):
        S.reset_state()
        st.rerun()
