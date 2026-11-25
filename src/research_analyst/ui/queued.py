"""
ui/queued.py — the planning and research phases when runs execute on a worker (``EXECUTION_MODE=queue``).

The UI only submits, polls, approves and cancels; it never runs the graph.  The run id is kept in the page URL
(``?run=...``), so reloading the page or restarting the UI reattaches to the run.  Anyone who has that link can see
the run: the app has no authentication.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import redis
import streamlit as st

from research_analyst.config import settings
from research_analyst.jobs import store as S
from research_analyst.safety.sanitize import sanitize_report_markdown
from research_analyst.ui import state as ui_state

LIVE_REFRESH_S = 0.7
RUN_PARAM = "run"
_UNREACHABLE = "**Cannot reach Redis**, which queue mode needs. Check `REDIS_URL` and that the server is running."


def enabled() -> bool:
    return settings.execution_mode == "queue"


@st.cache_resource
def _store() -> S.RunStore:
    from research_analyst.jobs.connection import new_client  # noqa: PLC0415
    return S.RunStore(new_client(decode=True))


@st.cache_resource
def _graph() -> Any:
    """The pipeline compiled against the shared checkpointer; the UI only READS state from it."""
    from research_analyst.jobs.checkpoint import RedisCheckpointSaver  # noqa: PLC0415
    from research_analyst.jobs.connection import new_client  # noqa: PLC0415
    from research_analyst.pipeline.graph import build_graph  # noqa: PLC0415
    saver = RedisCheckpointSaver(new_client(decode=False), prefix=settings.queue_namespace,
                                 ttl_seconds=settings.run_ttl_seconds)
    return build_graph(checkpointer=saver)


def get_store() -> S.RunStore:
    return _store()


def get_graph() -> Any:
    return _graph()


def _elapsed_s(record: dict[str, Any]) -> float | None:
    """Wall-clock time since research first started, across attempts (progress restarts when a run is resumed)."""
    if not record.get("started_at"):
        return None
    finished = float(record["finished_at"]) if record["status"] in S.TERMINAL else time.time()
    return max(0.0, finished - float(record["started_at"]))


# ── moving between phases ─────────────────────────────────────────────────────

def _show(record: dict[str, Any]) -> bool:
    """Switch the page to whatever the run's status calls for.  False while it is still queued or running."""
    status, run_id = record["status"], record["run_id"]
    if status == S.AWAITING_APPROVAL:
        snapshot = get_graph().get_state({"configurable": {"thread_id": run_id}})
        st.session_state.plan_result = {**snapshot.values, "__interrupt__": list(snapshot.interrupts)}
        st.session_state.pop("sq_textarea", None)
        st.session_state.phase = "awaiting_approval"
    elif status == S.DONE:
        snapshot = get_graph().get_state({"configurable": {"thread_id": run_id}})
        st.session_state.final_state = snapshot.values
        st.session_state.run_elapsed_s = _elapsed_s(record)
        st.session_state.phase = "done"
    elif status == S.FAILED:
        st.session_state.error = record.get("error") or "The run failed."
        st.session_state.phase = "error"
    elif status == S.CANCELLED:
        if record.get("usage"):
            from research_analyst.observability.usage import tracker  # noqa: PLC0415
            tracker.restore(run_id, record["usage"])
        st.session_state.phase = "cancelled"
    else:
        return False
    return True


def reattach() -> None:
    """A fresh browser session opened on ``?run=<id>`` picks that run up again."""
    run_id = st.query_params.get(RUN_PARAM)
    if not run_id or st.session_state.phase != "idle":
        return
    try:
        record = get_store().get(str(uuid.UUID(run_id)))
        if record is None:
            del st.query_params[RUN_PARAM]
            st.info("That research run no longer exists (runs expire after a while).")
            return
        st.session_state.update(session_id=record["run_id"], thread_id=record["run_id"], query=record["query"],
                                review_rounds=record["review_rounds"], submitted_run=record["run_id"])
        if not _show(record):
            st.session_state.phase = "planning" if record["stage"] == S.PLAN else "researching"
    except ValueError:                       # not a run id
        del st.query_params[RUN_PARAM]
    except redis.RedisError:
        st.error(_UNREACHABLE)


def approve(sub_questions: list[str]) -> None:
    try:
        started = get_store().approve(st.session_state.session_id, sub_questions)
    except redis.RedisError:
        st.error(_UNREACHABLE)
        return
    if not started:
        record = get_store().get(st.session_state.session_id)
        if record is None:
            st.error("This plan has expired. Start the research again.")
            return
        if _show(record):
            st.rerun()
    st.session_state.phase = "researching"
    st.rerun()


def cancel_run() -> None:
    """Button callback: ask the run to stop; the next poll shows the outcome."""
    try:
        get_store().cancel(st.session_state.get("session_id", ""))
    except redis.RedisError:
        st.error(_UNREACHABLE)


# ── phases ────────────────────────────────────────────────────────────────────

@st.fragment(run_every=LIVE_REFRESH_S)
def _live_panel() -> None:
    try:
        store = get_store()
        record = store.get(st.session_state.session_id)
        if record is None:
            st.session_state.update(phase="error", error="This run no longer exists (expired or removed).")
            st.rerun()
            return
        if _show(record):
            st.rerun()
            return
        workers = store.workers_alive()
    except redis.RedisError:
        st.error(_UNREACHABLE)
        return

    if not workers:
        st.warning("No worker is running, so this run is waiting. Start one with "
                   "`python -m research_analyst.jobs.worker` (Docker Compose starts it for you).")
    planning = record["stage"] == S.PLAN
    if record["status"] == S.QUEUED:
        st.progress(0.0, text="Waiting for a worker…")
        return
    if planning:
        st.progress(0.0, text="Generating the research plan — this takes around 20–60 seconds…")
        return
    snap = record.get("progress")
    if snap:
        mins, secs = divmod(int(_elapsed_s(record) or snap["elapsed_s"]), 60)
        st.progress(snap["fraction"], text=f"{int(snap['fraction'] * 100)}% — {snap['label']}")
        resumed = f" · attempt {record['attempts'] - 1}" if record["attempts"] > 2 else ""
        st.caption(f"{snap['branches_done']} of {snap['n_branches']} sub-questions researched · "
                   f"elapsed {mins}:{secs:02d}{resumed} · typically 3–6 minutes · percentage is approximate")
    else:
        st.progress(0.0, text="Starting…")
    if record.get("draft"):
        st.markdown('<div class="live-label">Writing the report — live draft</div>', unsafe_allow_html=True)
        with st.container(key="livedraft"):
            st.markdown(sanitize_report_markdown(record["draft"]) + " ▍")
        st.markdown('<div class="live-note">Raw model output as it arrives. Citation numbers are checked and the reference list is '
                    "built from the sources actually cited when the writer finishes; this text may change slightly.</div>",
                    unsafe_allow_html=True)


def _phase_waiting() -> None:
    st.markdown(f"**Query:** {st.session_state.query}")
    _live_panel()
    st.button("Stop research", key="stop_research", on_click=cancel_run,
              help="Cancels the run. Requests already sent to the model finish in the background.")
    st.caption("The research runs on a background worker. You can close this page and return to the same link.")


def phase_planning() -> None:
    run_id = st.session_state.session_id
    if st.session_state.get("submitted_run") != run_id:
        try:
            get_store().submit(run_id, st.session_state.query, st.session_state.get("review_rounds", 0))
        except redis.RedisError:
            st.error(_UNREACHABLE)
            if st.button("Start over"):
                ui_state.reset_state()
                st.rerun()
            return
        st.session_state.submitted_run = run_id
        st.query_params[RUN_PARAM] = run_id
    _phase_waiting()


def phase_researching() -> None:
    _phase_waiting()

