"""Results page."""
from __future__ import annotations

import streamlit as st

from research_analyst.grounding.verify import verify_payload
from research_analyst.observability.progress import progress

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
        with st.container(key="report"):
            st.markdown(sanitize_report_markdown(final_report), unsafe_allow_html=False)
        C.render_citation_stats(state)
    else:
        st.warning(
            "No final report was produced.  "
            "Check the agent logs under logs/agent_traces.jsonl.",
        )

    if final_report:
        payload = verify_payload(final_report, state.get("search_results", []), state.get("extracted_facts", []),
                                 state.get("references", []), state.get("claim_verdicts"))
        C.render_verify(payload["claims"], payload["evidence"], key="live", verdicts=payload["verdicts"])

    snap = progress.snapshot(st.session_state.get("session_id", ""))
    C.render_run_summary(state.get("usage_summary") or {},
                         snap["elapsed_s"] if snap else st.session_state.get("run_elapsed_s"))

    C.render_react_traces(state.get("react_traces") or [])

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
