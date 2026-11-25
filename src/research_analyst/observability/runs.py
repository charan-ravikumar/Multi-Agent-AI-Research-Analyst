"""
observability/runs.py — one durable record per research run, and the aggregation behind the Monitor page.

``build_record`` turns a finished (or failed / cancelled) run into a flat JSON-serialisable dict; ``append`` writes it as
one line of ``settings.runs_file``; ``load`` reads the newest N back; ``summarise`` computes the headline numbers.  The
last two are pure functions of the records, so they are unit-tested without Streamlit.

What a record deliberately does NOT contain: the report text or evidence (those are in the UI session / Langfuse), and
nothing secret.  The user's query is stored (truncated) because "which question failed" is the first thing you ask.
On Streamlit Cloud the filesystem is ephemeral, so the history there is only as long as the container lives.
"""
from __future__ import annotations

import json
import logging
import statistics
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_analyst.config import settings

_log = logging.getLogger(__name__)
_lock = threading.Lock()

COMPLETE, PARTIAL, FAILED, CANCELLED = "complete", "partial", "failed", "cancelled"


def _error_kind(error: str) -> str:
    """Coarse bucket for an error message so the page can count causes instead of listing raw strings."""
    e = error.lower()
    if "daily request limit" in e or "dailyquota" in e:
        return "daily quota"
    if any(k in e for k in ("429", "resource_exhausted", "rate limit", "quota")):
        return "rate limit"
    if any(k in e for k in ("503", "504", "unavailable", "deadline", "overload", "high demand")):
        return "model overloaded"
    if "circuit open" in e:
        return "circuit open"
    if "time limit" in e or "timeout" in e or "timed out" in e:
        return "timeout"
    if "all " in e and "branch" in e:
        return "all branches failed"
    if "budget" in e:
        return "token budget"
    return "other"


def build_record(
    session_id: str, status: str, state: dict | None, error: str, elapsed_s: float, query: str,
    usage: dict | None = None, extra: dict | None = None,
) -> dict[str, Any]:
    """Flatten one run.  ``status`` is complete/partial/failed/cancelled (see ``classify``)."""
    state = state or {}
    final = state.get("usage_summary") or {}
    # The finished run carries its own final summary (the in-memory tracker is cleared by the memory node); the tracker
    # snapshot is only the fallback for runs that died before producing one.
    usage = final if final.get("calls") else (usage or final)
    cs = state.get("citation_stats") or {}
    cv = state.get("claim_verdicts") or {}
    summ, judge = cv.get("summary") or {}, cv.get("judge") or {}
    traces = state.get("react_traces") or []
    gaps = state.get("gap_items") or []
    routing = [str(r) for r in state.get("routing_history", [])]
    from research_analyst.observability import prompt_store  # noqa: PLC0415
    rec: dict[str, Any] = {
        "run_id": session_id,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "query": (query or state.get("query") or "")[:200],
        "status": status,
        "error": error[:300], "error_kind": _error_kind(error) if error else "",
        "elapsed_s": round(float(elapsed_s), 1),
        "tokens": int(usage.get("total_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "calls": int(usage.get("calls") or 0),
        "truncated_calls": int(usage.get("truncated_calls") or 0),
        "cost_usd": usage.get("estimated_cost_usd"),
        "by_agent": {a: {k: v.get(k, 0) for k in ("calls", "total_tokens", "latency_s")} for a, v in (usage.get("by_agent") or {}).items()},
        "by_model": {m: {k: v.get(k, 0) for k in ("calls", "total_tokens", "latency_s")} for m, v in (usage.get("by_model") or {}).items()},
        "by_key": {k: v.get("calls", 0) for k, v in (usage.get("by_key") or {}).items()},
        "cited_ratio": cs.get("report_cited_ratio", cs.get("cited_ratio")),
        "report_claims": cs.get("report_claim_sentences"),
        "sources_cited": cs.get("sources_cited"), "sources_consulted": cs.get("sources_consulted"),
        "invalid_markers_removed": cs.get("invalid_markers_removed"),
        "judge": {"status": judge.get("status"), "judged": summ.get("judged", 0), "supported": summ.get("supported", 0),
                  "partial": summ.get("partial", 0), "unsupported": summ.get("unsupported", 0)},
        "branch_failures": sum(1 for r in routing if r.startswith("branch_failed")),
        "coverage_warnings": sum(1 for g in gaps if g.get("kind") == "coverage"),
        "open_questions": sum(1 for g in gaps if g.get("kind") == "open_question"),
        "contradictions": len(state.get("contradictions", [])),
        "sub_questions": len(state.get("sub_questions", [])),
        "words": len((state.get("final_report") or "").split()),
        "search_mode": settings.search_mode,
        "react": {"steps": sum(len(t.get("steps", [])) for t in traces),
                  "stopped": dict(Counter(t.get("stopped", "?") for t in traces))} if traces else None,
        "models": {"fast": settings.gemini_fast_model, "strong": settings.gemini_strong_model,
                   "strong_agents": sorted(settings.strong_agents)},
        "review_rounds": int(state.get("review_rounds") or 0),
        "prompts": prompt_store.used(),
    }
    if extra:
        rec.update(extra)
    return rec


def classify(state: dict | None, error: str, cancelled: bool) -> str:
    if cancelled:
        return CANCELLED
    if error or not state:
        return FAILED
    failed = any(str(r).startswith("branch_failed") for r in state.get("routing_history", []))
    return PARTIAL if failed else COMPLETE


def append(record: dict[str, Any], path: str | None = None) -> None:
    """Append one record; never raises (monitoring must not break a run)."""
    target = Path(path or settings.runs_file)
    try:
        with _lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not write run record: %s", exc)


def load(limit: int = 500, path: str | None = None) -> list[dict[str, Any]]:
    """Newest ``limit`` records, oldest first.  Unreadable lines are skipped."""
    target = Path(path or settings.runs_file)
    try:
        lines = target.read_text("utf-8").splitlines()[-limit:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers for the Monitor page.  Cancelled runs are excluded from success/latency (the user chose to stop)."""
    counted = [r for r in records if r.get("status") != CANCELLED]
    n = len(counted)
    ok = [r for r in counted if r["status"] in (COMPLETE, PARTIAL)]
    times = [r["elapsed_s"] for r in ok if r.get("elapsed_s")]
    judged = sum(r.get("judge", {}).get("judged", 0) for r in ok)
    agent_tokens: Counter = Counter()
    agent_lat: Counter = Counter()
    for r in ok:
        for a, v in (r.get("by_agent") or {}).items():
            agent_tokens[a] += v.get("total_tokens", 0)
            agent_lat[a] += v.get("latency_s", 0.0)
    cited = [r["cited_ratio"] for r in ok if isinstance(r.get("cited_ratio"), (int, float))]
    return {
        "runs": len(records), "counted": n,
        "complete": sum(1 for r in counted if r["status"] == COMPLETE),
        "partial": sum(1 for r in counted if r["status"] == PARTIAL),
        "failed": sum(1 for r in counted if r["status"] == FAILED),
        "cancelled": len(records) - n,
        "success_rate": round(len(ok) / n, 3) if n else None,
        "clean_rate": round(sum(1 for r in counted if r["status"] == COMPLETE) / n, 3) if n else None,
        "p50_s": _pct(times, 0.5), "p95_s": _pct(times, 0.95),
        "median_tokens": int(statistics.median([r["tokens"] for r in ok])) if ok else None,
        "median_calls": int(statistics.median([r["calls"] for r in ok])) if ok else None,
        "tokens_total": sum(r["tokens"] for r in records),
        "cited_ratio": round(statistics.mean(cited), 3) if cited else None,
        "judged_claims": judged,
        "supported_share": round(sum(r["judge"]["supported"] for r in ok if r.get("judge")) / judged, 3) if judged else None,
        "partial_share": round(sum(r["judge"]["partial"] for r in ok if r.get("judge")) / judged, 3) if judged else None,
        "unsupported_share": round(sum(r["judge"]["unsupported"] for r in ok if r.get("judge")) / judged, 3) if judged else None,
        "errors": dict(Counter(r["error_kind"] for r in counted if r["status"] == FAILED and r.get("error_kind"))),
        "agent_tokens": dict(agent_tokens), "agent_latency_s": {a: round(v, 1) for a, v in agent_lat.items()},
        "truncated_calls": sum(r.get("truncated_calls", 0) for r in ok),
        "branch_failures": sum(r.get("branch_failures", 0) for r in ok),
        "invalid_markers_removed": sum(r.get("invalid_markers_removed") or 0 for r in ok),
    }
