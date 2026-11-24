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
            "**Groq rate limit reached.**\n\n"
            "The free tier enforces a tokens-per-minute (TPM) and a daily "
            "tokens-per-day (TPD) budget.  Wait a few minutes for a TPM reset, "
            "or up to 24 hours for the rolling TPD window to recover.",
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
