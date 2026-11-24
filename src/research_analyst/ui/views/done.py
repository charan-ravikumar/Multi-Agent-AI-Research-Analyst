"""Results page."""
from __future__ import annotations


import streamlit as st

from research_analyst.ui import components as C
from research_analyst.ui import state as S


def phase_done() -> None:
    state = st.session_state.final_state

    st.markdown(f"**Query:** {state.get('query', st.session_state.query)}")

    # Debate-loop outcome banner (must appear before the report so the reader
    # sees the disclosure context before reading the content)
    C.render_debate_banner(state)
    C.render_degraded_banner(state)

    # Final report
    final_report = state.get("final_report", "")
    if final_report:
        from research_analyst.safety.sanitize import sanitize_report_markdown  # noqa: PLC0415
        st.markdown(sanitize_report_markdown(final_report), unsafe_allow_html=False)
        C.render_citation_stats(state)
    else:
        st.warning(
            "No final report was produced.  "
            "Check the agent logs under logs/agent_traces.jsonl.",
            icon="⚠",
        )

    C.render_usage(state)

    # Gaps and contradictions — the backend explicitly surfaces these;
    # the UI must not hide them.
    C.render_gaps_and_contradictions(state)

    # Citations
    C.render_citations(state.get("citations", []))

    # Routing history (debug / transparency)
    C.render_routing_history(state)

    st.divider()
    if st.button("New research"):
        S.reset_state()
        st.rerun()
