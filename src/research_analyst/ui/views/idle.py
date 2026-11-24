"""Start page: topic input, example questions, key check."""
from __future__ import annotations

import uuid

import streamlit as st

from research_analyst.ui.about import SAMPLE_QUESTIONS


def set_query(q: str) -> None:
    st.session_state["query_input"] = q


def missing_provider_keys() -> list[str]:
    from research_analyst.config import settings  # noqa: PLC0415
    return [name for name, val in (("GEMINI_API_KEY", settings.gemini_api_key),) if not val]


def phase_idle() -> None:
    st.markdown("### What do you want to research?")
    missing = missing_provider_keys()
    if missing:
        st.error(
            "**No LLM API key is configured**, so research cannot run. In Streamlit Cloud open "
            "*App → Settings → Secrets* and add `GEMINI_API_KEY = \"...\"` as a plain top-level entry, "
            "then reboot the app.",
            icon="🔑",
        )
    blocked = bool(missing)
    from research_analyst.llm import preflight  # noqa: PLC0415
    level, message = preflight.describe(preflight.check_models())
    if level == "error":
        st.error(message, icon="🚫")
        blocked = True
    elif level == "warning":
        st.warning(message, icon="⚠")
    query = st.text_input(
        "Research topic",
        placeholder="e.g. Applications of large language models in healthcare",
        key="query_input",
    )
    st.checkbox(
        "Run the critic/reviser review loop (opt-in)",
        key="review_loop",
        help="Measured on 3 test queries: more citation markers, no faithfulness gain, slightly lower "
             "completeness, ~75% more tokens. Off by default.",
    )
    st.caption("Try an example:")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for n, (col, q) in enumerate(zip(cols, SAMPLE_QUESTIONS, strict=True)):
        col.button(q.split("?")[0][:48] + "…", key=f"sample_{n}", help=q, use_container_width=True,
                   on_click=set_query, args=(q,))
    submitted = st.button(
        "Start research →",
        type="primary",
        disabled=not (query or "").strip() or blocked,
    )
    clean_query = None
    if submitted:
        try:
            from research_analyst.safety.sanitize import validate_query  # noqa: PLC0415
            clean_query = validate_query(query)
        except ValueError as exc:
            st.error(str(exc))
    if clean_query:
        st.session_state.phase      = "planning"
        st.session_state.review_rounds = 2 if st.session_state.get("review_loop") else 0
        st.session_state.query      = clean_query
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.thread_id  = str(uuid.uuid4())
        st.rerun()
