"""Research phase: live progress while the background thread runs."""
from __future__ import annotations

import threading

import streamlit as st

from research_analyst.observability.progress import progress
from research_analyst.safety.sanitize import sanitize_report_markdown
from research_analyst.ui import state as S
from research_analyst.ui.trace_tail import read_trace_tail


LIVE_REFRESH_S = 0.7      # the live panel re-renders by itself (no full-page rerun, so no flicker)


def _elapsed(snap: dict) -> str:
    mins, secs = divmod(int(snap["elapsed_s"]), 60)
    return f"{mins}:{secs:02d}"


@st.fragment(run_every=LIVE_REFRESH_S)
def _live_panel() -> None:
    """Progress, the answer as it is being written, and the activity log.  Reruns on its own while the run is alive."""
    holder: dict = st.session_state.bg_result
    thread: threading.Thread = st.session_state.bg_thread
    if holder.get("done") or not thread.is_alive():
        st.rerun()                                   # whole app: phase_researching sees the finished run and moves on
        return
    snap = progress.snapshot(st.session_state.session_id)
    if snap:
        st.progress(snap["fraction"], text=f"{int(snap['fraction'] * 100)}% — {snap['label']}")
        st.caption(f"{snap['branches_done']} of {snap['n_branches']} sub-questions researched · elapsed {_elapsed(snap)} · "
                   "typically 3–6 minutes · percentage is approximate")
        if snap.get("stream"):
            st.markdown('<div class="live-label">Writing the report — live draft</div>', unsafe_allow_html=True)
            with st.container(key="livedraft"):
                st.markdown(sanitize_report_markdown(snap["stream"]) + " ▍")
            st.markdown('<div class="live-note">Raw model output as it arrives. Citation numbers are checked and the reference list is '
                        "built from the sources actually cited when the writer finishes; this text may change slightly.</div>",
                        unsafe_allow_html=True)
    else:
        st.progress(0.0, text="Starting…")

    events = read_trace_tail(st.session_state.session_id)
    with st.expander("Technical activity log", expanded=False):
        for ev in reversed(events):
            lvl = (ev.get("level") or "info").upper()
            tag = "ERROR" if lvl == "ERROR" else "WARN" if lvl == "WARNING" else "INFO"
            st.write(" ".join(p for p in (f"`{tag}`", f"`{ev.get('agent') or ''}`", f"`{ev.get('step') or ''}`",
                                          (ev.get("message") or "")[:120]) if p))
        if not events:
            st.write("No events yet.")


def phase_researching() -> None:
    result_holder: dict          = st.session_state.bg_result
    thread:        threading.Thread = st.session_state.bg_thread

    if not result_holder.get("done") and thread.is_alive():
        st.markdown(f"**Query:** {st.session_state.query}")
        _live_panel()
        st.button("Stop research", key="stop_research", on_click=S.cancel_run,
                  help="Cancels the run. Requests already sent to the model finish in the background.")
        return

    # ── thread finished ────────────────────────────────────────────────────
    if result_holder.get("cancelled"):
        st.session_state.phase = "cancelled"
        st.rerun()
        return

    if result_holder.get("error"):
        st.session_state.phase = "error"
        st.session_state.error = result_holder["error"]
        st.rerun()
        return

    st.session_state.final_state = result_holder.get("result", {})
    st.session_state.phase       = "done"
    progress.finish(st.session_state.session_id)
    st.rerun()
