"""Monitor tab: how the system is behaving across runs — reliability, speed, cost, quota and answer quality.

Everything shown comes from two sources: the per-run records in ``settings.runs_file`` (observability/runs.py) and the
process-wide counters of the LLM client (observability/metrics.py).  The page only reads; it never calls an LLM.
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from research_analyst.config import settings
from research_analyst.llm import client as llm_client
from research_analyst.observability import metrics, runs
from research_analyst.ui import theme

STATUS_COLORS = {"complete": theme.OK, "partial": theme.WARN, "failed": theme.ACCENT, "cancelled": "#9a9384"}
VERDICT_COLORS = {"supported": theme.OK, "partial": theme.WARN, "unsupported": theme.ACCENT}


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _dur(s: float | None) -> str:
    if s is None:
        return "—"
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def _style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(background="transparent", font="Hanken Grotesk")
            .configure_axis(labelColor=theme.MUTED, titleColor=theme.MUTED, gridColor=theme.RULE, domainColor=theme.INK,
                            tickColor=theme.INK, labelFont="JetBrains Mono", titleFont="JetBrains Mono", labelFontSize=10, titleFontSize=10)
            .configure_legend(labelColor=theme.INK, titleColor=theme.MUTED, labelFont="Hanken Grotesk", titleFont="JetBrains Mono")
            .configure_view(stroke=None))


def _quota_bars() -> None:
    for q in llm_client.quota_status():
        rpd = q["rpd"]
        used = q["used"]
        frac = min(1.0, used / rpd) if rpd else 0.0
        cls = "hot" if frac >= 0.9 else "warn" if frac >= 0.7 else ""
        limit = f"{used} of {rpd} requests today" if rpd else f"{used} requests today (no daily limit set)"
        split = " + ".join(f"{k}: {n}" for k, n in q.get("keys", {}).items()) if len(q.get("keys", {})) > 1 else ""
        extra = " · ".join(x for x in (f"{q['rpm']}/min" if q["rpm"] else "", f"{q['tpm']:,} tokens/min" if q["tpm"] else "",
                                        f"split across keys — {split}" if split else "") if x)
        role = []
        if q["model"] == settings.gemini_fast_model:
            role.append("fast")
        if q["model"] == settings.gemini_strong_model:
            role.append("strong: " + (", ".join(sorted(settings.strong_agents)) or "no agents"))
        st.markdown(
            f'<div class="quota"><div class="quota-head"><span>{q["model"]}'
            f'{" · " + " / ".join(role) if role else ""}</span><span>{limit}</span></div>'
            f'<div class="quota-bar"><i class="{cls}" style="width:{frac * 100:.1f}%"></i></div>'
            f'<div class="live-note">{extra or "&nbsp;"}</div></div>', unsafe_allow_html=True)


def _status_chart(summary: dict) -> None:
    df = pd.DataFrame({"status": ["complete", "partial", "failed"],
                       "runs": [summary["complete"], summary["partial"], summary["failed"]]})
    df = df[df["runs"] > 0]
    if df.empty:
        return
    chart = alt.Chart(df).mark_bar(size=26).encode(
        x=alt.X("runs:Q", stack="zero", title=None, axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["status", "runs"]).properties(height=70)
    st.altair_chart(_style(chart), use_container_width=True)


def _time_chart(df: pd.DataFrame) -> None:
    chart = alt.Chart(df).mark_bar(size=14).encode(
        x=alt.X("n:O", title="run (oldest → newest)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("minutes:Q", title="minutes"),
        color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())), legend=None),
        tooltip=["when", "status", "minutes", "tokens", "query"]).properties(height=220)
    st.altair_chart(_style(chart), use_container_width=True)


def _bar(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    chart = alt.Chart(df).mark_bar(color=theme.INK, size=14).encode(
        y=alt.Y(f"{y}:N", sort="-x", title=None), x=alt.X(f"{x}:Q", title=title), tooltip=[y, x]).properties(height=max(90, 34 * len(df)))
    st.altair_chart(_style(chart), use_container_width=True)


def _verdict_chart(records: list[dict]) -> None:
    rows = []
    for i, r in enumerate(records, 1):
        j = r.get("judge") or {}
        for v in ("supported", "partial", "unsupported"):
            if j.get("judged"):
                rows.append({"n": i, "verdict": v, "claims": j.get(v, 0), "query": r["query"][:60]})
    if not rows:
        st.caption("No run has been judged yet (the grounding judge writes these).")
        return
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar(size=14).encode(
        x=alt.X("n:O", title="run (oldest → newest)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("claims:Q", stack="normalize", title="claims", axis=alt.Axis(format="%")),
        color=alt.Color("verdict:N", scale=alt.Scale(domain=list(VERDICT_COLORS), range=list(VERDICT_COLORS.values())), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["query", "verdict", "claims"]).properties(height=220)
    st.altair_chart(_style(chart), use_container_width=True)


def _counters() -> None:
    err = metrics.by_label("llm_error", "status")
    cols = st.columns(4)
    cols[0].metric("LLM calls ok", f"{int(metrics.total('llm_success')):,}")
    cols[1].metric("Retries", f"{int(metrics.total('llm_retry')):,}")
    cols[2].metric("Waiting on limits", f"{int(metrics.total('rate_limit_wait_seconds'))} s")
    cols[3].metric("Circuit opens", f"{int(metrics.total('breaker_open'))}")
    if err:
        st.markdown("".join(f'<span class="pill bad">{k} × {int(v)}</span>' for k, v in sorted(err.items())), unsafe_allow_html=True)
    retried = metrics.by_label("structured_retry", "agent")
    if retried:
        st.caption("Structured answers that needed a corrective re-ask: " + ", ".join(f"{a} ×{int(n)}" for a, n in sorted(retried.items())))
    where = " In queue mode the model calls happen in the worker, so these stay at zero here." if settings.execution_mode == "queue" else ""
    st.caption(f"Counters cover this process only ({int(metrics.uptime_s() // 60)} min since start) and reset on restart.{where}")


def render_monitor() -> None:
    theme.kicker("Operations")
    st.markdown("### Monitor")
    st.caption("How the system behaves across runs: is it reliable, how long and how much does a run take, "
               "how much of today's free quota is left, and do the claims hold up against their sources.")
    records = runs.load(500)
    s = runs.summarise(records)

    cols = st.columns(6)
    cols[0].metric("Runs recorded", s["runs"], help="Includes runs you stopped.")
    cols[1].metric("Completed", _pct(s["success_rate"]), help="Share of runs that produced a report (complete or partial). Stopped runs are excluded.")
    cols[2].metric("Median time", _dur(s["p50_s"]), delta=f"p95 {_dur(s['p95_s'])}", delta_color="off")
    cols[3].metric("Median tokens", f"{s['median_tokens']:,}" if s["median_tokens"] else "—", delta=f"{s['median_calls']} calls" if s["median_calls"] else None, delta_color="off")
    cols[4].metric("Claims supported", _pct(s["supported_share"]), delta=f"{s['judged_claims']} judged" if s["judged_claims"] else None, delta_color="off",
                   help="Share of cited claims the LLM judge found supported by their cited passages.")
    cols[5].metric("Claims cited", _pct(s["cited_ratio"]), help="Average share of claim sentences that carry a citation marker (coverage, not truth).")

    st.markdown('<div class="kicker">Free-tier quota</div>', unsafe_allow_html=True)
    _quota_bars()
    st.caption("Daily counts are stored on disk and reset at midnight Pacific time, like Google's own quota. Every retry counts. "
               "With two API keys the limits shown are the sum of both accounts; the wrapper splits calls between them.")

    st.markdown('<div class="kicker">Reliability</div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        _status_chart(s)
        st.caption(f"{s['branch_failures']} sub-question(s) failed inside otherwise finished runs · "
                   f"{s['truncated_calls']} answer(s) cut off at the token limit · {s['invalid_markers_removed']} invented citation marker(s) removed.")
    with right:
        if s["errors"]:
            st.markdown("**Why runs failed**")
            st.markdown("".join(f'<span class="pill bad">{k} × {n}</span>' for k, n in s["errors"].items()), unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill ok">no failed runs</span>' if records else '<span class="pill off">no data yet</span>', unsafe_allow_html=True)
    _counters()

    if not records:
        st.info("No runs have been recorded yet. Finish a research run on the Research tab and its numbers appear here.")
        return

    recent = records[-30:]
    df = pd.DataFrame([{"n": i, "when": r["ts"][5:16].replace("T", " "), "status": r["status"], "minutes": round(r["elapsed_s"] / 60, 2),
                        "tokens": r["tokens"], "query": r["query"][:60]} for i, r in enumerate(recent, 1)])
    st.markdown('<div class="kicker">Speed</div>', unsafe_allow_html=True)
    _time_chart(df)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="kicker">Where the tokens go</div>', unsafe_allow_html=True)
        if s["agent_tokens"]:
            _bar(pd.DataFrame({"agent": list(s["agent_tokens"]), "tokens": list(s["agent_tokens"].values())}), "tokens", "agent", "tokens (all recorded runs)")
    with b:
        st.markdown('<div class="kicker">Where the time goes</div>', unsafe_allow_html=True)
        if s["agent_latency_s"]:
            _bar(pd.DataFrame({"agent": list(s["agent_latency_s"]), "seconds": list(s["agent_latency_s"].values())}), "seconds", "agent", "model time, seconds (summed over calls)")

    st.markdown('<div class="kicker">Do the claims hold up?</div>', unsafe_allow_html=True)
    _verdict_chart(recent)

    st.markdown('<div class="kicker">Recent runs</div>', unsafe_allow_html=True)
    table = pd.DataFrame([{
        "when (UTC)": r["ts"].replace("T", " ")[:16], "status": r["status"], "question": r["query"], "time": _dur(r["elapsed_s"]),
        "tokens": r["tokens"], "calls": r["calls"], "supported": _pct((r["judge"]["supported"] / r["judge"]["judged"]) if r.get("judge", {}).get("judged") else None),
        "cited": _pct(r.get("cited_ratio")), "search": r.get("search_mode", ""), "error": r.get("error_kind", "")} for r in reversed(recent)])
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download all records (JSONL)", "\n".join(json.dumps(r) for r in records), file_name="runs.jsonl", mime="application/jsonl")
