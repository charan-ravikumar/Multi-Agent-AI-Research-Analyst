"""Plan approval: editable sub-questions, then start the research thread."""
from __future__ import annotations

import threading

import streamlit as st

from research_analyst.observability.progress import progress
from research_analyst.ui import components as C
from research_analyst.ui import state as S
from research_analyst.ui.runner import research_thread


def phase_awaiting_approval() -> None:
    result    = st.session_state.plan_result
    interrupt = result.get("__interrupt__", [])
    iv        = interrupt[0].value if interrupt else {}

    st.subheader("Review the proposed research plan")
    st.markdown(f"**Query:** {result.get('query', st.session_state.query)}")

    # Planner's strategy notes
    rp    = result.get("research_plan")
    notes = C.attr(rp, "strategy_notes", "")
    depth = C.attr(rp, "depth", "standard")
    if notes:
        st.caption(f"Planner strategy ({depth}): _{notes}_")

    # Cross-run memory: related past reports (from Redis; absent if Redis is down)
    related = iv.get("related_past_reports") or C.str_list(result.get("related_past_reports"))
    if related and isinstance(related, list) and isinstance(related[0], dict):
        C.render_related_past_reports(related)
    elif related:
        # related_past_reports comes as a list of dicts from the interrupt value
        pass

    st.divider()
    st.markdown(
        "**Edit the sub-questions** below — one per line.  "
        "Delete, reorder, or add questions freely before approving."
    )

    # Pre-populate the textarea on first visit to this phase
    sqs = C.str_list(result.get("sub_questions") or iv.get("sub_questions"))
    if "sq_textarea" not in st.session_state:
        st.session_state.sq_textarea = "\n".join(sqs)

    st.text_area(
        "Sub-questions (one per line)",
        key="sq_textarea",
        height=max(160, 44 * max(len(sqs), 3)),
    )

    col_approve, col_cancel, *_ = st.columns([2, 2, 6])
    with col_approve:
        if st.button("Approve plan →", type="primary"):
            raw_text = st.session_state.sq_textarea or ""
            approved = [s.strip() for s in raw_text.split("\n") if s.strip()]
            if not approved:
                st.error("Add at least one sub-question before approving.")
                return
            # Clean up textarea state before thread starts
            st.session_state.pop("sq_textarea", None)
            start_research(approved)

    with col_cancel:
        if st.button("Cancel / start over"):
            S.reset_state()
            st.rerun()


def start_research(approved_sqs: list[str]) -> None:
    """Kick off the research background thread and transition to 'researching'."""
    cfg           = {"configurable": {"thread_id": st.session_state.thread_id,
                                      "session_id": st.session_state.session_id}}
    result_holder = {"done": False}
    progress.start(st.session_state.session_id, len(approved_sqs), st.session_state.get("review_rounds", 0))
    t = threading.Thread(
        target=research_thread,
        args=(S.get_graph(), cfg, approved_sqs, result_holder),
        daemon=True,
    )
    t.start()
    st.session_state.bg_thread = t
    st.session_state.bg_result = result_holder
    st.session_state.phase     = "researching"
    st.rerun()
