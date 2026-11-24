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
sys.path.insert(0, str(ROOT / "src"))   # Streamlit Cloud does not install the project itself

os.environ.setdefault("SEARCH_MAX_RESULTS", "5")
os.environ.setdefault("READER_MAX_CONCURRENT_LLM_CALLS", "2")

# ── stdlib ─────────────────────────────────────────────────────────────────────
import asyncio
import json
import re
import threading
import time
import uuid

# ── third-party ────────────────────────────────────────────────────────────────
import streamlit as st

from research_analyst.ui.about import ABOUT_MD, SAMPLE_QUESTIONS  # noqa: E402
from research_analyst.ui.secrets_env import export_secrets_to_env  # noqa: E402
from research_analyst.grounding.gaps import curate_gaps  # noqa: E402
from research_analyst.observability.progress import progress  # noqa: E402

# Streamlit Cloud secrets -> environment, BEFORE config.Settings is first imported (it is imported lazily
# below).  Handles secrets pasted under a [section] header or in lower case.  Local runs have no
# secrets.toml; that raises and is ignored (a local .env is used instead).
try:
    export_secrets_to_env(dict(st.secrets), os.environ)
except Exception:  # noqa: BLE001 — no secrets configured
    pass

# ── constants ──────────────────────────────────────────────────────────────────
LOG_FILE = ROOT / "logs" / "agent_traces.jsonl"
POLL_INTERVAL = 3   # seconds between progress refreshes during research phase
TRACE_TAIL = 10     # number of recent trace events to show


# ── lazy project imports (cached so module init runs exactly once) ─────────────

@st.cache_resource
def _load_graph():
    """Import and cache the compiled LangGraph app across all Streamlit reruns."""
    from research_analyst.pipeline.graph import app  # noqa: PLC0415
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

def _read_trace_tail(session_id: str, n: int = TRACE_TAIL, max_bytes: int = 256_000) -> list[dict]:
    """Return the last *n* log entries for *session_id*.

    Reads only the final ``max_bytes`` of the shared trace file (not the whole file) so the
    cost of a poll does not grow with the log; a partial first line is discarded."""
    if not LOG_FILE.exists():
        return []
    entries: list[dict] = []
    try:
        with open(LOG_FILE, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            chunk = fh.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()
        if size > max_bytes:
            lines = lines[1:]                      # first line is probably cut in half
        for raw in lines:
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if rec.get("session_id") == session_id:
                entries.append(rec)
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
        from research_analyst.config import settings  # noqa: PLC0415
        from research_analyst.observability import tracing  # noqa: PLC0415

        async def _run():
            sid = cfg.get("configurable", {}).get("session_id", "")
            with tracing.run_span("research", sid):
                return await asyncio.wait_for(
                    app_ref.ainvoke(Command(resume=resume_value), config=cfg),
                    timeout=settings.run_timeout_seconds,
                )

        result = loop.run_until_complete(_run())
        result_holder["result"] = result
    except asyncio.TimeoutError:
        result_holder["error"] = "Run exceeded the time limit and was cancelled."
    except Exception as exc:
        result_holder["error"] = str(exc)
    finally:
        result_holder["done"] = True
        loop.close()


# ── render helpers ─────────────────────────────────────────────────────────────

def _render_debate_banner(state: dict) -> None:
    forced  = state.get("debate_forced", False)
    rounds  = state.get("debate_round", 0)
    if state.get("debate_review_skipped"):
        st.warning(
            "**The Critic review was skipped** (its output could not be parsed), so this "
            "draft was NOT independently reviewed.",
            icon="⚠",
        )
    elif forced:
        st.warning(
            f"**Debate loop hit the round cap** ({rounds} round(s) completed).  "
            "The Critic still had unresolved objections when the limit was reached.  "
            "Unresolved gaps and contradictions are disclosed in the sections below.",
            icon="🔁",
        )
    elif not rounds and not state.get("review_rounds"):
        st.caption("Review loop was off for this run (enable it on the start page).")
    elif rounds:
        st.success(
            f"Critic accepted the revised draft after {rounds} debate round(s).",
            icon="✅",
        )


def _render_degraded_banner(state: dict) -> None:
    failed = [e for e in _str_list(state.get("routing_history")) if e.startswith("branch_failed")]
    if failed:
        st.warning(
            f"**Partial report:** {len(failed)} research branch(es) failed and were skipped. "
            "The affected sub-questions are listed under Unresolved gaps.",
            icon="⚠",
        )


def _render_citation_stats(state: dict) -> None:
    s = state.get("citation_stats") or {}
    if not s or not s.get("claim_sentences"):
        return
    st.caption(
        f"Grounding check (automated, lexical — not a proof of correctness): "
        f"{s.get('cited', 0)}/{s['claim_sentences']} claim sentences cite a source; "
        f"{s.get('uncited', 0)} uncited, {s.get('weakly_supported', 0)} weakly supported "
        f"(share <2 key terms with the cited facts); "
        f"{s.get('invalid_markers_removed', 0)} invalid citation marker(s) removed; "
        f"{s.get('sources_cited', 0)} of {s.get('sources_consulted', 0)} consulted sources cited."
    )


def _render_usage(state: dict) -> None:
    u = state.get("usage_summary") or {}
    if not u.get("calls"):
        return
    cost = u.get("estimated_cost_usd")
    with st.expander(
        f"Run usage: {u['total_tokens']:,} tokens in {u['calls']} LLM calls"
        + (f" (≈ ${cost:.4f})" if cost is not None else ""),
        expanded=False,
    ):
        st.json({"by_agent": u.get("by_agent"), "by_model": u.get("by_model"),
                 "truncated_calls": u.get("truncated_calls"), "budget": u.get("budget") or "off"})


def _start_followup(question: str) -> None:
    """Button callback: start over with a follow-up question prefilled (the user still presses Start)."""
    base = (st.session_state.get("query") or "").strip()
    text = f"{base} — {question}" if base and question.lower() not in base.lower() else question
    from research_analyst.config import settings  # noqa: PLC0415
    _reset_state()
    st.session_state["query_input"] = text[: settings.max_query_chars]


def _render_gaps_and_contradictions(state: dict) -> None:
    """Coverage warnings (system-detected), follow-up research (model-suggested), contradictions."""
    items = state.get("gap_items") or []
    curated: dict = curate_gaps(items) if items else {"coverage": [], "open_questions": [], "dropped": 0}
    contras = _str_list(state.get("contradictions"))
    legacy = [] if items else _str_list(state.get("unresolved_gaps"))
    if not (curated["coverage"] or curated["open_questions"] or contras or legacy):
        return
    st.divider()

    if curated["coverage"]:
        st.subheader("⚠ Coverage warnings")
        st.caption("Detected by the system while researching. They affect how far to trust the matching sections.")
        for c in curated["coverage"]:
            st.warning(c["text"])

    if curated["open_questions"]:
        st.subheader("🔎 Suggested follow-up research")
        st.caption(
            "Questions the gathered sources did not answer, suggested by the model "
            "(unverified, deduplicated"
            + (f"; {curated['dropped']} similar or lower-priority ones hidden" if curated["dropped"] else "")
            + "). Click one to start a new research run on it."
        )
        for n, o in enumerate(curated["open_questions"]):
            left, right = st.columns([6, 1])
            left.markdown(f"**{o['text']}**  \n<span style='opacity:.6'>from: {o['sub_question'][:90]}</span>",
                          unsafe_allow_html=True)
            right.button("Research this", key=f"followup_{n}", on_click=_start_followup, args=(o["text"],))

    if legacy:
        with st.expander(f"Gaps noted ({len(legacy)})"):
            for g in legacy:
                st.write(f"- {g}")

    if contras:
        with st.expander(f"⚡ Contradictions between sources ({len(contras)})", expanded=True):
            for c in contras:
                st.info(c)


def _render_citations(citations: list) -> None:
    if not citations:
        return
    st.subheader(f"Sources consulted ({len(citations)})")
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

def _set_query(q: str) -> None:
    st.session_state["query_input"] = q


def _missing_provider_keys() -> list[str]:
    from research_analyst.config import settings  # noqa: PLC0415
    return [name for name, val in (("GROQ_API_KEY", settings.groq_api_key),
                                   ("GEMINI_API_KEY", settings.gemini_api_key)) if not val]


def _phase_idle() -> None:
    st.markdown("### What do you want to research?")
    missing = _missing_provider_keys()
    if len(missing) == 2:
        st.error(
            "**No LLM API key is configured**, so research cannot run. In Streamlit Cloud open "
            "*App → Settings → Secrets* and add `GROQ_API_KEY = \"...\"` (and optionally "
            "`GEMINI_API_KEY`) as plain top-level entries, then reboot the app.",
            icon="🔑",
        )
    elif missing:
        st.warning(f"{missing[0]} is not set; the fallback provider is unavailable.", icon="⚠")
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
                   on_click=_set_query, args=(q,))
    submitted = st.button(
        "Start research →",
        type="primary",
        disabled=not (query or "").strip() or len(missing) == 2,
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


# ── phase: planning ────────────────────────────────────────────────────────────

def _phase_planning() -> None:
    """
    Blocking invocation of the graph until it hits the plan-approval interrupt.
    asyncio.run() is safe here: Streamlit does not hold a running event loop
    between script reruns (verified against 1.58.0).
    """
    from research_analyst.models import ResearchState  # noqa: PLC0415

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
            gap_items=[],
            draft=[],
            final_report="",
            iteration=0,
            routing_history=[],
            auto_approve_plan=False,   # human checkpoint must fire
            review_rounds=st.session_state.get("review_rounds", 0),
        )
        cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            from research_analyst.observability import tracing  # noqa: PLC0415

            async def _plan():
                with tracing.run_span("planning", st.session_state.session_id, st.session_state.query):
                    return await _get_graph().ainvoke(initial, config=cfg)

            result = asyncio.run(_plan())
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
    cfg           = {"configurable": {"thread_id": st.session_state.thread_id,
                                      "session_id": st.session_state.session_id}}
    result_holder = {"done": False}
    progress.start(st.session_state.session_id, len(approved_sqs), st.session_state.get("review_rounds", 0))
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

        events = _read_trace_tail(st.session_state.session_id)
        with st.expander("Technical activity log", expanded=False):
            for ev in reversed(events):
                lvl = (ev.get("level") or "info").upper()
                icon = "🔴" if lvl == "ERROR" else "🟡" if lvl == "WARNING" else "🟢"
                st.write(" ".join(p for p in (icon, f"`{ev.get('agent') or ''}`", f"`{ev.get('step') or ''}`",
                                              (ev.get("message") or "")[:120]) if p))
            if not events:
                st.write("No events yet.")

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
    progress.finish(st.session_state.session_id)
    st.rerun()


# ── phase: done ────────────────────────────────────────────────────────────────

def _phase_done() -> None:
    state = st.session_state.final_state

    st.markdown(f"**Query:** {state.get('query', st.session_state.query)}")

    # Debate-loop outcome banner (must appear before the report so the reader
    # sees the disclosure context before reading the content)
    _render_debate_banner(state)
    _render_degraded_banner(state)

    # Final report
    final_report = state.get("final_report", "")
    if final_report:
        from research_analyst.safety.sanitize import sanitize_report_markdown  # noqa: PLC0415
        st.markdown(sanitize_report_markdown(final_report), unsafe_allow_html=False)
        _render_citation_stats(state)
    else:
        st.warning(
            "No final report was produced.  "
            "Check the agent logs under logs/agent_traces.jsonl.",
            icon="⚠",
        )

    _render_usage(state)

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
        _reset_state()
        st.rerun()


# ── state reset ────────────────────────────────────────────────────────────────

_SESSION_KEYS = (
    "phase", "query", "session_id", "thread_id",
    "plan_result", "final_state",
    "bg_thread", "bg_result",
    "error",
    "sq_textarea", "_sq_default",
    "query_input", "review_loop", "review_rounds",
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
        page_title="AI Research Analyst",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 AI Research Analyst")
    st.caption("Ask a question — get a sourced report. Sub-questions, search, reading and writing run in parallel.")

    if "phase" not in st.session_state:
        st.session_state.phase = "idle"

    tab_research, tab_about = st.tabs(["🔬 Research", "ℹ️ About this app"])
    with tab_about:
        st.markdown(ABOUT_MD)
    with tab_research:
        handler = _PHASE_DISPATCH.get(st.session_state.phase)
        if handler is None:
            st.error(f"Unknown phase: {st.session_state.phase!r}")
            _reset_state()
            st.rerun()
            return
        handler()


if __name__ == "__main__":
    main()
