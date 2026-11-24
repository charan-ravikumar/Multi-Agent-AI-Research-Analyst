"""Error page with provider-aware messages."""
from __future__ import annotations

import re

import streamlit as st

from research_analyst.ui import state as S


def phase_error() -> None:
    err = st.session_state.get("error", "Unknown error")
    err_lower = err.lower()

    if re.search(r"\b(rate[- ]?limit\w*|429|tpm|tpd|exhausted|quota)\b", err_lower):
        st.error(
            "**Gemini quota reached.**\n\n"
            "The free tier limits requests per minute (RPM) and per day (RPD) for each model. "
            "Wait a minute for an RPM reset; a daily limit resets at midnight Pacific time.",
            icon="🚦",
        )
    elif any(kw in err_lower for kw in ("redis", "10061", "connection refused")):
        st.error(
            "**Redis connection refused.**  "
            "Cross-run memory is unavailable, but research can proceed without it.  "
            "Start the run again — the backend will gracefully skip memory lookups.",
            icon="🔌",
        )
    else:
        st.error(f"**Research pipeline failed:**\n\n{err}", icon="❌")

    with st.expander("Full error detail"):
        st.code(err, language=None)

    if st.button("Start over"):
        S.reset_state()
        st.rerun()
