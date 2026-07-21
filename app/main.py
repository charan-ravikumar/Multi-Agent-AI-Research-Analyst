"""
app/main.py — Streamlit UI for the Agentic Research & Knowledge Intelligence Platform.

Async strategy
--------------
Phase 1 — planning (START → planner → plan_approval → interrupt):
  asyncio.run() called in the Streamlit main thread, blocking.
  Safe because Streamlit does not maintain a persistent event loop between
  script reruns; asyncio.run() creates and destroys its own loop each time.
  Verified in tools/_verify_async.py.  Duration ~20–60 s; spinner shown.

Phase 2 — research (Command(resume=approved_sqs) → completion):
  threading.Thread with asyncio.new_event_loop() so it never fights the main
  thread's event-loop lifecycle.  Streamlit polls the thread state every
  POLL_INTERVAL seconds via time.sleep + st.rerun(), showing live progress
  from logs/agent_traces.jsonl for the current session_id.

State machine (st.session_state.phase)
---------------------------------------
  "idle"              → topic input form
  "planning"          → blocking ainvoke until interrupt; → "awaiting_approval"
  "awaiting_approval" → editable plan review; → "researching" or reset
  "researching"       → background thread + live progress poll; → "done" / "error"
  "done"              → final report, citations, gaps, contradictions, debate info
  "error"             → error display with rate-limit / Redis / generic handling
"""

# ── env vars MUST be set before any project imports ───────────────────────────
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("SEARCH_MAX_RESULTS", "5")
os.environ.setdefault("READER_MAX_CONCURRENT_LLM_CALLS", "2")

# ── stdlib ─────────────────────────────────────────────────────────────────────
import asyncio
import json
import threading
import time
import uuid

# ── third-party ────────────────────────────────────────────────────────────────
import streamlit as st

# ── constants ──────────────────────────────────────────────────────────────────
LOG_FILE = ROOT / "logs" / "agent_traces.jsonl"
POLL_INTERVAL = 3   # seconds between progress refreshes during research phase
TRACE_TAIL = 10     # number of recent trace events to show


# ── lazy project imports (cached so module init runs exactly once) ─────────────

@st.cache_resource
def _load_graph():
    """Import and cache the compiled LangGraph app across all Streamlit reruns."""
    from orchestrator.graph import app  # noqa: PLC0415
    return app


def _get_graph():
    return _load_graph()


# ── generic helpers ────────────────────────────────────────────────────────────

def _attr(obj, key: str, default=None):
    """Safe attribute / key access on both Pydantic models and plain dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _str_list(val) -> list[str]:
    """Return *val* coerced to a list[str], handling None and Pydantic models."""
    if not val:
        return []
    if isinstance(val, str):
        return [val]
    return [str(v) for v in val]


# ── trace-log helpers ──────────────────────────────────────────────────────────

def _read_trace_tail(session_id: str, n: int = TRACE_TAIL) -> list[dict]:
    """Return the last *n* log entries for *session_id* from agent_traces.jsonl."""
    if not LOG_FILE.exists():
        return []
    entries: list[dict] = []
    try:
        with open(LOG_FILE, encoding="utf-8") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                    if rec.get("session_id") == session_id:
                        entries.append(rec)
                except Exception:
                    pass
    except OSError:
        pass
    return entries[-n:]


def _progress_label(events: list[dict]) -> str:
    """Derive a short human-readable status from the most recent trace entry."""
    if not events:
        return "Starting agents…"
    latest = events[-1]
    agent = latest.get("agent") or ""
    step  = latest.get("step")  or ""
    msg   = latest.get("message") or ""
    parts = [p for p in (agent, step) if p]
    head  = " › ".join(parts) if parts else ""
    tail  = f": {msg[:80]}" if msg else ""
    return head + tail if (head or tail) else "Running…"


# ── background thread (research phase) ────────────────────────────────────────

def _research_thread(
    app_ref,
    cfg: dict,
    resume_value: list[str],
    result_holder: dict,
) -> None:
    """
    Run the graph to completion inside a dedicated asyncio event loop.

    Stores outcome into *result_holder*:
      • success  → {"result": <state dict>, "done": True}
      • failure  → {"error": <str>,         "done": True}
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from langgraph.types import Command  # noqa: PLC0415
        result = loop.run_until_complete(
            app_ref.ainvoke(Command(resume=resume_value), config=cfg)
        )
        result_holder["result"] = result
    except Exception as exc:
        result_holder["error"] = str(exc)
    finally:
        result_holder["done"] = True
        loop.close()


# ── render helpers ─────────────────────────────────────────────────────────────

def _render_debate_banner(state: dict) -> None:
    forced  = state.get("debate_forced", False)
    rounds  = state.get("debate_round", 0)
    if forced:
        st.warning(
            f"**Debate loop hit the round cap** ({rounds} round(s) completed).  "
            "The Critic still had unresolved objections when the limit was reached.  "
            "Unresolved gaps and contradictions are disclosed in the sections below.",
            icon="🔁",
        )
    elif rounds:
        st.success(
            f"Critic accepted the revised draft after {rounds} debate round(s).",
            icon="✅",
        )


def _render_gaps_and_contradictions(state: dict) -> None:
    gaps    = _str_list(state.get("unresolved_gaps"))
    contras = _str_list(state.get("contradictions"))
    if not gaps and not contras:
        return
    st.divider()
    left, right = st.columns(2)
    with left:
        if gaps:
            st.subheader("⚠ Unresolved gaps")
            for g in gaps:
                st.warning(g)
    with right:
        if contras:
            st.subheader("⚡ Contradictions flagged")
            for c in contras:
                st.info(c)


def _render_citations(citations: list) -> None:
    if not citations:
        return
    st.subheader(f"Citations ({len(citations)})")
    seen: set[str] = set()
    for i, cit in enumerate(citations, 1):
        url     = _attr(cit, "url",         "#") or "#"
        title   = _attr(cit, "title",       "Untitled") or "Untitled"
        snippet = _attr(cit, "snippet",     "")  or ""
        src     = _attr(cit, "source_type", "web") or "web"
        if url in seen:
            continue
        seen.add(url)
        with st.expander(f"{i}. {title}  `[{src}]`"):
            if snippet:
                st.write(snippet)
            st.markdown(f"[{url}]({url})")


def _render_related_past_reports(related: list) -> None:
    if not related:
        return
    with st.expander(f"Related past research ({len(related)} report(s) found in memory)"):
        for r in related[:5]:
            ts     = r.get("timestamp_human") or r.get("timestamp") or "?"
            q      = r.get("query", "Unknown query")[:120]
            tokens = r.get("_overlap_tokens", [])
            st.markdown(
                f"- **[{ts}]** {q}  \n"
                f"  _Shared topics: {', '.join(tokens) if tokens else 'none detected'}_"
            )


def _render_routing_history(state: dict) -> None:
    routing = _str_list(state.get("routing_history"))
    if not routing:
        return
    with st.expander("Routing history (debug)", expanded=False):
        for entry in routing:
            st.code(entry, language=None)


# ── phase: idle ───────────────────────────────────────────────────────────────

def _phase_idle() -> None:
    st.markdown("### What do you want to research?")
    query = st.text_input(
        "Research topic",
        placeholder="e.g. Applications of large language models in healthcare",
        key="query_input",
    )
    submitted = st.button(
        "Start research →",
        type="primary",
        disabled=not (query or "").strip(),
    )
    if submitted and query.strip():
        st.session_state.phase      = "planning"
        st.session_state.query      = query.strip()
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.thread_id  = str(uuid.uuid4())
        st.rerun()


# ── phase: planning ────────────────────────────────────────────────────────────

def _phase_planning() -> None:
    """
    Blocking invocation of the graph until it hits the plan-approval interrupt.
    asyncio.run() is safe here: Streamlit does not hold a running event loop
    between script reruns (verified against 1.58.0).
    """
    from models import ResearchState  # noqa: PLC0415

    st.markdown(f"**Query:** {st.session_state.query}")

    with st.spinner("Generating research plan — this takes around 20–60 seconds…"):
        initial = ResearchState(
            session_id=st.session_state.session_id,
            query=st.session_state.query,
            research_plan=None,
            sub_questions=[],
            search_results=[],
            failed_sub_questions=[],
            extracted_facts=[],
            citations=[],
            contradictions=[],
            unresolved_gaps=[],
            draft=[],
            critique=None,
            final_report="",
            iteration=0,
            routing_history=[],
            auto_approve_plan=False,   # human checkpoint must fire
        )
        cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            result = asyncio.run(_get_graph().ainvoke(initial, config=cfg))
        except Exception as exc:
            st.session_state.phase = "error"
            st.session_state.error = str(exc)
            st.rerun()
            return

    interrupted = result.get("__interrupt__")
    if not interrupted:
        # Graph ran to completion without pausing (should not happen with
        # auto_approve_plan=False, but handle it gracefully).
        st.session_state.phase       = "done"
        st.session_state.final_state = result
        st.rerun()
        return

    st.session_state.plan_result = result
    st.session_state.phase       = "awaiting_approval"
    st.rerun()


# ── phase: awaiting_approval ──────────────────────────────────────────────────

def _phase_awaiting_approval() -> None:
    result    = st.session_state.plan_result
    interrupt = result.get("__interrupt__", [])
    iv        = interrupt[0].value if interrupt else {}

    st.subheader("Review the proposed research plan")
    st.markdown(f"**Query:** {result.get('query', st.session_state.query)}")

    # Planner's strategy notes
    rp    = result.get("research_plan")
    notes = _attr(rp, "strategy_notes", "")
    depth = _attr(rp, "depth", "standard")
    if notes:
        st.caption(f"Planner strategy ({depth}): _{notes}_")

    # Cross-run memory: related past reports (from Redis; absent if Redis is down)
    related = iv.get("related_past_reports") or _str_list(result.get("related_past_reports"))
    if related and isinstance(related, list) and isinstance(related[0], dict):
        _render_related_past_reports(related)
    elif related:
        # related_past_reports comes as a list of dicts from the interrupt value
        pass

    st.divider()
    st.markdown(
        "**Edit the sub-questions** below — one per line.  "
        "Delete, reorder, or add questions freely before approving."
    )

    # Pre-populate the textarea on first visit to this phase
    sqs = _str_list(result.get("sub_questions") or iv.get("sub_questions"))
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
            _start_research(approved)

    with col_cancel:
        if st.button("Cancel / start over"):
            _reset_state()
            st.rerun()


def _start_research(approved_sqs: list[str]) -> None:
    """Kick off the research background thread and transition to 'researching'."""
    cfg           = {"configurable": {"thread_id": st.session_state.thread_id}}
    result_holder = {"done": False}
    t = threading.Thread(
        target=_research_thread,
        args=(_get_graph(), cfg, approved_sqs, result_holder),
        daemon=True,
    )
    t.start()
    st.session_state.bg_thread = t
    st.session_state.bg_result = result_holder
    st.session_state.phase     = "researching"
    st.rerun()


# ── phase: researching ─────────────────────────────────────────────────────────

def _phase_researching() -> None:
    result_holder: dict          = st.session_state.bg_result
    thread:        threading.Thread = st.session_state.bg_thread

    if not result_holder.get("done") and thread.is_alive():
        # ── show live progress ─────────────────────────────────────────────
        st.markdown(f"**Query:** {st.session_state.query}")
        events = _read_trace_tail(st.session_state.session_id)
        label  = _progress_label(events)

        with st.status(f"Researching… — {label}", expanded=True):
            if events:
                st.markdown("**Recent agent activity**")
                for ev in reversed(events):  # newest at top
                    lvl   = (ev.get("level") or "info").upper()
                    agent = ev.get("agent") or ""
                    step  = ev.get("step")  or ""
                    msg   = ev.get("message") or ""
                    icon  = "🔴" if lvl == "ERROR" else "🟡" if lvl == "WARNING" else "🟢"
                    badge = f"`{agent}`" if agent else ""
                    tag   = f"`{step}`"  if step  else ""
                    line  = " ".join(p for p in (icon, badge, tag, msg[:120]) if p)
                    st.write(line)
            else:
                st.write("Starting agents — first trace events will appear shortly…")

        # Poll: wait then trigger next rerun
        time.sleep(POLL_INTERVAL)
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
    st.rerun()


# ── phase: done ────────────────────────────────────────────────────────────────

def _phase_done() -> None:
    state = st.session_state.final_state

    st.markdown(f"**Query:** {state.get('query', st.session_state.query)}")

    # Debate-loop outcome banner (must appear before the report so the reader
    # sees the disclosure context before reading the content)
    _render_debate_banner(state)

    # Final report
    final_report = state.get("final_report", "")
    if final_report:
        st.markdown(final_report, unsafe_allow_html=False)
    else:
        st.warning(
            "No final report was produced.  "
            "Check the agent logs under logs/agent_traces.jsonl.",
            icon="⚠",
        )

    # Gaps and contradictions — the backend explicitly surfaces these;
    # the UI must not hide them.
    _render_gaps_and_contradictions(state)

    # Citations
    _render_citations(state.get("citations", []))

    # Routing history (debug / transparency)
    _render_routing_history(state)

    st.divider()
    if st.button("New research"):
        _reset_state()
        st.rerun()


# ── phase: error ──────────────────────────────────────────────────────────────

def _phase_error() -> None:
    err = st.session_state.get("error", "Unknown error")
    err_lower = err.lower()

    if any(kw in err_lower for kw in ("rate", "tpm", "tpd", "exhausted", "quota")):
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
        _reset_state()
        st.rerun()


# ── state reset ────────────────────────────────────────────────────────────────

_SESSION_KEYS = (
    "phase", "query", "session_id", "thread_id",
    "plan_result", "final_state",
    "bg_thread", "bg_result",
    "error",
    "sq_textarea", "_sq_default",
    "query_input",
)


def _reset_state() -> None:
    """Clear all per-run session state.  The cached graph (@st.cache_resource)
    is intentionally preserved — no need to re-import on each run."""
    for key in _SESSION_KEYS:
        st.session_state.pop(key, None)


# ── main entry point ──────────────────────────────────────────────────────────

_PHASE_DISPATCH = {
    "idle":              _phase_idle,
    "planning":          _phase_planning,
    "awaiting_approval": _phase_awaiting_approval,
    "researching":       _phase_researching,
    "done":              _phase_done,
    "error":             _phase_error,
}


def main() -> None:
    st.set_page_config(
        page_title="Agentic Research Platform",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Agentic Research & Knowledge Intelligence")

    if "phase" not in st.session_state:
        st.session_state.phase = "idle"

    handler = _PHASE_DISPATCH.get(st.session_state.phase)
    if handler is None:
        st.error(f"Unknown phase: {st.session_state.phase!r}")
        _reset_state()
        st.rerun()
        return

    handler()


if __name__ == "__main__":
    main()
