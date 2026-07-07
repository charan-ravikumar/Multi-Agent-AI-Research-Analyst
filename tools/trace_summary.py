"""
tools/trace_summary.py — Reconstruct a human-readable timeline for any run.

Reads logs/agent_traces.jsonl and prints every decision point for a given
session in chronological order.  No external services needed.

Usage
-----
    # Most recent run
    python tools/trace_summary.py --last

    # N-th most recent run (--last 2 = second-most-recent)
    python tools/trace_summary.py --last 2

    # Exact session UUID
    python tools/trace_summary.py --session-id <uuid>

    # Different log file
    python tools/trace_summary.py --last --log-file path/to/agent_traces.jsonl

What it shows
-------------
  - Every agent invocation: start time, latency, LLM token counts
  - Research plan: query + all sub-questions
  - Per-branch outcomes: search result count, facts extracted, draft sections
  - Retry attempts (zero-results retries)
  - Plan approval: auto vs human, related past reports found
  - Debate rounds: objection counts, revision rounds, organic/forced resolution
  - Memory write: report persistence to Redis
  - LLM cost summary: calls, total tokens, total LLM time, models used
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
DEFAULT_LOG = ROOT / "logs" / "agent_traces.jsonl"

# ── Step display config ───────────────────────────────────────────────────────
# Maps step name → (display label, whether to show latency, whether to show tokens)
# Steps not in this table are shown with their raw step name.

_STEP_LABELS: dict[str, str] = {
    "agent_start":        "START",
    "agent_complete":     "DONE ",
    "agent_error":        "ERROR",
    "llm_call":           "LLM  ",
    "plan_complete":      "PLAN ",
    "plan_validate":      "PLAN?",
    "branch_start":       "BRAN>",
    "branch_end":         "BRAN<",
    "search_start":       "SRCH>",
    "search_done":        "SRCH<",
    "search_empty":       "SRCH0",
    "synth_start":        "SYN> ",
    "synth_done":         "SYN< ",
    "writer_start":       "WRT> ",
    "writer_validate":    "WRT? ",
    "writer_error":       "WRT! ",
    "memory_lookup":      "MEM? ",
    "memory_lookup_fail": "MEM! ",
    "memory_write":       "MEM+ ",
    "memory_write_fail":  "MEM! ",
    "debate_round":       "DEB  ",
    "debate_resolved":    "DEB✓ ",
    "debate_forced":      "DEB! ",
}

# Steps shown in a more muted way (INFO level but routine bookkeeping)
_QUIET_STEPS = {"agent_start", "agent_complete", "search_start", "synth_start"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def _fmt_ts(ts_str: str) -> str:
    dt = _parse_ts(ts_str)
    return dt.strftime("%H:%M:%S.%f")[:-3]  # trim to milliseconds


def _fmt_lat(v: float | None) -> str:
    if v is None:
        return ""
    return f"  [{v:.2f}s]"


def _fmt_tok(t: dict | None) -> str:
    if not t:
        return ""
    p = t.get("prompt_tokens", 0)
    c = t.get("completion_tokens", 0)
    model = t.get("model", "")
    return f"  tok={p}+{c}={p+c}" + (f" ({model})" if model else "")


def _interesting_extra(extra: dict) -> str:
    """Pull the most useful fields from the extra dict into a short string."""
    parts = []
    # Ordered by importance for the trace display
    for key in (
        "sub_questions", "sub_question_count", "depth",
        "results", "facts", "sections",
        "attempt", "error",
        "report_id",
    ):
        val = extra.get(key)
        if val is None:
            continue
        if key == "sub_questions":
            # Don't inline the list here — shown in the plan header
            pass
        elif key == "error":
            # Truncate long error messages
            parts.append(f"err={str(val)[:60]}")
        else:
            parts.append(f"{key}={val}")
    return "  " + "  ".join(parts) if parts else ""


# ── Load records ──────────────────────────────────────────────────────────────

def _load_records(log_file: Path, session_id: str) -> list[dict]:
    records: list[dict] = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("session_id") == session_id:
                records.append(rec)
    records.sort(key=lambda r: r["timestamp"])
    return records


def _find_session_ids(log_file: Path) -> list[str]:
    """Return unique session_ids sorted by first-seen timestamp (oldest first)."""
    seen: dict[str, str] = {}  # sid -> first timestamp
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            sid = rec.get("session_id", "")
            if sid and sid not in seen:
                seen[sid] = rec["timestamp"]
    return sorted(seen.keys(), key=lambda s: seen[s])


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_timeline(records: list[dict], session_id: str) -> str:
    if not records:
        return f"No records found for session_id={session_id}"

    lines: list[str] = []
    t_start = _parse_ts(records[0]["timestamp"])
    t_end   = _parse_ts(records[-1]["timestamp"])
    total_s = (t_end - t_start).total_seconds()

    # ── Extract plan info from log records ───────────────────────────────────
    query = ""
    sub_questions: list[str] = []
    for r in records:
        step = r.get("step", "")
        extra = r.get("extra", {})
        if step == "plan_complete":
            sub_questions = extra.get("sub_questions", [])
            # query might be in the message or extra
        if step == "search_start" and not query:
            # fall back: use searcher session start to infer presence
            pass
        if r.get("agent") == "planner" and step == "agent_start":
            pass

    # Try to get query from writer_start message
    for r in records:
        if r.get("step") == "writer_start":
            # "writing final report: ..." doesn't have query directly
            pass

    # ── Header ───────────────────────────────────────────────────────────────
    W = 74
    lines.append("=" * W)
    lines.append(f"  RUN TIMELINE  —  session_id={session_id}")
    lines.append("=" * W)

    if sub_questions:
        lines.append(f"  Plan  : {len(sub_questions)} sub-question(s)")
        for i, sq in enumerate(sub_questions, 1):
            lines.append(f"          {i}. {sq}")
    lines.append(f"  Start : {_fmt_ts(records[0]['timestamp'])}")
    lines.append(f"  End   : {_fmt_ts(records[-1]['timestamp'])}")
    lines.append(f"  Elapsed: {total_s:.1f}s")
    lines.append("")

    # Column headers
    COL_TIME   = 12
    COL_AGENT  = 11
    COL_STEP   = 7
    lines.append(
        f"  {'TIME':<{COL_TIME}}  {'AGENT':<{COL_AGENT}}  {'':>{COL_STEP}}  DETAIL"
    )
    lines.append("  " + "─" * (W - 2))

    for r in records:
        ts    = _fmt_ts(r["timestamp"])
        agent = r.get("agent", "?")
        step  = r.get("step", "")
        msg   = r.get("message", "")
        lat   = r.get("latency_s")
        tok   = r.get("tokens")
        level = r.get("level", "INFO")
        extra = r.get("extra", {})

        label = _STEP_LABELS.get(step, step[:7])
        quiet = step in _QUIET_STEPS

        # Level prefix: nothing for INFO quiet, '· ' for INFO important,
        # '! ' for WARNING, 'E ' for ERROR
        if level == "WARNING":
            pfx = "! "
        elif level in ("ERROR", "CRITICAL"):
            pfx = "E "
        elif quiet:
            pfx = "  "
        else:
            pfx = "· "

        detail = pfx + msg + _fmt_lat(lat) + _fmt_tok(tok) + _interesting_extra(extra)

        lines.append(
            f"  {ts:<{COL_TIME}}  {agent:<{COL_AGENT}}  {label:>{COL_STEP}}  {detail}"
        )

    # ── LLM summary ──────────────────────────────────────────────────────────
    llm_records = [r for r in records if r.get("step") == "llm_call"]
    if llm_records:
        lines.append("")
        lines.append("  " + "─" * (W - 2))
        total_tok   = sum((r.get("tokens") or {}).get("total_tokens", 0) for r in llm_records)
        total_llm_s = sum(r.get("latency_s") or 0 for r in llm_records)
        # Model breakdown
        models: dict[str, int] = {}
        for r in llm_records:
            m = (r.get("tokens") or {}).get("model", "unknown")
            models[m] = models.get(m, 0) + 1

        lines.append(f"  LLM CALLS    : {len(llm_records)}")
        lines.append(f"  TOTAL TOKENS : {total_tok:,}")
        lines.append(f"  LLM WALL TIME: {total_llm_s:.1f}s  ({100*total_llm_s/total_s:.0f}% of run)")
        for model, count in sorted(models.items(), key=lambda x: -x[1]):
            lines.append(f"    {count}× {model}")

    lines.append("=" * W)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a human-readable run timeline from agent_traces.jsonl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-id",
        metavar="UUID",
        help="Exact session UUID to inspect.",
    )
    group.add_argument(
        "--last",
        nargs="?",
        const=1,
        type=int,
        metavar="N",
        help="N-th most recent session (default: 1 = most recent).",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG),
        metavar="PATH",
        help=f"Path to agent_traces.jsonl (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all session IDs found in the log file and exit.",
    )
    args = parser.parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)

    if args.list_sessions:
        sids = _find_session_ids(log_file)
        if not sids:
            print("No sessions found.")
        else:
            print(f"Found {len(sids)} session(s) in {log_file}:")
            for i, sid in enumerate(reversed(sids), 1):
                print(f"  [{i} most recent] {sid}")
        return

    if args.session_id:
        session_id = args.session_id
    else:
        sids = _find_session_ids(log_file)
        if not sids:
            print("No sessions found in log file.", file=sys.stderr)
            sys.exit(1)
        n = min(args.last, len(sids))
        session_id = sids[-n]  # -1 = most recent, -2 = second most recent, etc.
        print(f"Session: {session_id}  (#{n} most recent)\n")

    records = _load_records(log_file, session_id)
    print(render_timeline(records, session_id))


if __name__ == "__main__":
    main()
