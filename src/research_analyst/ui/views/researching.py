"""Research phase: live progress while the background thread runs."""
from __future__ import annotations

import threading
import time

import streamlit as st

from research_analyst.observability.progress import progress
from research_analyst.ui import state as S
from research_analyst.ui.trace_tail import read_trace_tail


def phase_researching() -> None:
    result_holder: dict          = st.session_state.bg_result
    thread:        threading.Thread = st.session_state.bg_thread

    if not result_holder.get("done") and thread.is_alive():
        # ── show live progress ─────────────────────────────────────────────
        st.markdown(f"**Query:** {st.session_state.query}")
        snap = progress.snapshot(st.session_state.session_id)
        if snap:
            pct = int(snap["fraction"] * 100)
            st.progress(snap["fraction"], text=f"{pct}% — {snap['label']}")
            mins, secs = divmod(int(snap["elapsed_s"]), 60)
            st.caption(
                f"{snap['branches_done']} of {snap['n_branches']} sub-questions researched · "
                f"elapsed {mins}:{secs:02d} · typically 3–6 minutes · percentage is approximate"
            )
        else:
            st.progress(0.0, text="Starting…")

        events = read_trace_tail(st.session_state.session_id)
        with st.expander("Technical activity log", expanded=False):
            for ev in reversed(events):
                lvl = (ev.get("level") or "info").upper()
                icon = "🔴" if lvl == "ERROR" else "🟡" if lvl == "WARNING" else "🟢"
                st.write(" ".join(p for p in (icon, f"`{ev.get('agent') or ''}`", f"`{ev.get('step') or ''}`",
                                              (ev.get("message") or "")[:120]) if p))
            if not events:
                st.write("No events yet.")

        # Poll: wait then trigger next rerun
        time.sleep(S.POLL_INTERVAL)
        st.rerun()
        return

    # ── thread finished ────────────────────────────────────────────────────
    if result_holder.get("error"):
        st.session_state.phase = "error"
        st.session_state.error = result_holder["error"]
        st.rerun()
        return

    st.session_state.final_state = result_holder.get("result", {})
    st.session_state.phase       = "done"
    progress.finish(st.session_state.session_id)
    st.rerun()
