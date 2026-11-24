"""Reusable rendering pieces for the results page."""
from __future__ import annotations

from typing import Any

import streamlit as st

from research_analyst.grounding.gaps import curate_gaps
from research_analyst.ui import state as S


def attr(obj: Any, key: str, default: Any = None) -> Any:
    """Safe attribute / key access on both Pydantic models and plain dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def str_list(val: Any) -> list[str]:
    """Return *val* coerced to a list[str], handling None and Pydantic models."""
    if not val:
        return []
    if isinstance(val, str):
        return [val]
    return [str(v) for v in val]


def render_debate_banner(state: dict) -> None:
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


def render_degraded_banner(state: dict) -> None:
    failed = [e for e in str_list(state.get("routing_history")) if e.startswith("branch_failed")]
    if failed:
        st.warning(
            f"**Partial report:** {len(failed)} research branch(es) failed and were skipped. "
            "The affected sub-questions are listed under Unresolved gaps.",
            icon="⚠",
        )


def render_citation_stats(state: dict) -> None:
    s = state.get("citation_stats") or {}
    if not s:
        return
    if s.get("report_claim_sentences"):                    # measured on the FINAL report text
        head = (f"Citation check (automated — counts markers, not truth): "
                f"{s.get('report_cited', 0)}/{s['report_claim_sentences']} sentences of the report cite a source")
    elif s.get("claim_sentences"):
        head = (f"Citation check (automated — counts markers, not truth): "
                f"{s.get('cited', 0)}/{s['claim_sentences']} draft claim sentences cite a source")
    else:
        return
    st.caption(
        f"{head}; {s.get('uncited', 0)} uncited and {s.get('weakly_supported', 0)} weakly supported in the draft "
        f"(share <2 key terms with the cited facts); {s.get('invalid_markers_removed', 0)} invalid marker(s) removed; "
        f"{s.get('sources_cited', 0)} of {s.get('sources_consulted', 0)} consulted sources cited."
    )


def render_run_summary(usage: dict, elapsed_s: float | None = None, key: str = "live") -> None:
    """Time, tokens, LLM calls and cost of one run, plus a per-agent / per-model breakdown."""
    if not usage or not usage.get("calls"):
        return
    cost = usage.get("estimated_cost_usd")
    c1, c2, c3, c4 = st.columns(4)
    if elapsed_s:
        mins, secs = divmod(int(elapsed_s), 60)
        c1.metric("Time", f"{mins}:{secs:02d}")
    c2.metric("Tokens", f"{usage['total_tokens']:,}")
    c3.metric("LLM calls", f"{usage['calls']}")
    if cost is not None:
        c4.metric("Estimated cost", f"${cost:.4f}", help="Estimated from the configured LLM_PRICE_PER_1M_TOKENS table.")
    else:
        c4.metric("Cost", "$0 (free tier)", help="This deployment runs on free-tier API keys. Tokens are counted; set "
                  "LLM_PRICE_PER_1M_TOKENS (USD per 1M input/output tokens, per model) to see a paid-rate estimate.")
    with st.expander("Usage breakdown", expanded=False):
        st.json({"by_agent": usage.get("by_agent"), "by_model": usage.get("by_model"),
                 "truncated_calls": usage.get("truncated_calls"), "token_budget": usage.get("budget") or "off"},
                expanded=2)


def _select_claim(sel_key: str, claim_id: int) -> None:
    st.session_state[sel_key] = claim_id


_VERDICT_ICON = {"supported": "🟢", "partial": "🟡", "unsupported": "🔴"}


def render_judge_summary(verdicts: dict | None) -> None:
    """One-line result of the LLM grounding judge (or why it is absent)."""
    if not verdicts:
        return
    j, s = verdicts.get("judge") or {}, verdicts.get("summary") or {}
    if s.get("judged"):
        st.caption(f"LLM grounding judge ({j.get('model', '?')}): of {s['judged']} cited claims checked against their sources, "
                   f"{s['supported']} supported, {s['partial']} partial, {s['unsupported']} unsupported."
                   + (f" ({j['error']})" if j.get("error") else "")
                   + " The judge reads only the gathered passages and is itself a model: a triage signal, not proof.")
    elif j.get("status") in ("failed", "partial"):
        st.caption(f"LLM grounding judge unavailable ({j.get('error') or j.get('status')}); showing word-overlap labels instead.")


def render_react_traces(traces: list[dict]) -> None:
    """What the ReAct search agent thought and did, per sub-question (only present when SEARCH_MODE=react)."""
    if not traces:
        return
    with st.expander(f"Search agent reasoning (ReAct) — {len(traces)} sub-question(s)", expanded=False):
        for t in traces:
            st.markdown(f"**{t['sub_question']}** — stopped: `{t['stopped']}`")
            for step in t["steps"]:
                st.markdown(f"{step['step']}. *{step['thought'] or '…'}*  \n`{step['action']}({step['input']})`")


def render_verify(claims: list[dict], evidence: dict, key: str = "live", verdicts: dict | None = None) -> None:
    """Click a claim sentence -> the LLM judge's verdict plus the exact source passage(s) it rests on."""
    from research_analyst.grounding.verify import Claim, support_for  # noqa: PLC0415
    if not claims:
        return
    st.subheader("🔍 Verify the claims")
    st.caption("Click a sentence from the report to see the passage in the cited source(s) it rests on and whether an "
               "LLM judge found the claim supported by them. Highlighted words are shared between claim and source. "
               "Open the source to be sure.")
    render_judge_summary(verdicts)
    by_id = (verdicts or {}).get("verdicts") or {}
    sel_key = f"verify_sel_{key}"
    selected = st.session_state.get(sel_key, claims[0]["id"])
    left, right = st.columns([5, 6])
    with left, st.container(height=520, border=False):
        for c in claims:
            text = c["text"]
            st.button(f"{c['id']}. {text[:118]}{'…' if len(text) > 118 else ''}", key=f"{key}_claim_{c['id']}",
                      type="primary" if c["id"] == selected else "secondary", use_container_width=True,
                      on_click=_select_claim, args=(sel_key, c["id"]))
    with right:
        claim = next((c for c in claims if c["id"] == selected), claims[0])
        st.markdown(f"**Claim {claim['id']}:** {claim['text']}")
        icons = {"strong": "🟢", "partial": "🟡", "weak": "🔴"}
        verdict = by_id.get(str(claim["id"]))
        if verdict:
            note = f" — {verdict['reason']}" if verdict.get("reason") else ""
            st.markdown(f"{_VERDICT_ICON[verdict['verdict']]} **Judge: {verdict['verdict']}**{note}")
        else:
            st.caption("No LLM verdict for this claim; the label below is word overlap only.")
        for s in support_for(Claim(claim["id"], claim["text"], tuple(claim["refs"])), evidence):
            with st.container(border=True):
                title = f"[{s['ref']}] [{s['title']}]({s['url']})" if s["url"] else f"[{s['ref']}] {s['title']}"
                backs = verdict and s["ref"] in (verdict.get("supporting_refs") or [])
                overlap = f"word overlap: {s['label']} ({int(s['score'] * 100)}% of key terms)"
                st.markdown(f"**{title}**  \n" + (f"✅ judge: this source backs the claim · {overlap}" if backs
                                                 else f"{icons[s['label']]} {overlap}"))
                if s["passage_html"]:
                    st.markdown(f"<div style='font-size:0.92rem;line-height:1.5'>{s['passage_html']}</div>", unsafe_allow_html=True)
                else:
                    st.caption("No passage text was kept for this source.")
                if s["fact_html"]:
                    st.markdown(f"<div style='font-size:0.85rem;opacity:.75'>Extracted fact: {s['fact_html']}</div>", unsafe_allow_html=True)


def start_followup(question: str) -> None:
    """Button callback: start over with a follow-up question prefilled (the user still presses Start)."""
    base = (st.session_state.get("query") or "").strip()
    text = f"{base} — {question}" if base and question.lower() not in base.lower() else question
    from research_analyst.config import settings  # noqa: PLC0415
    S.reset_state()
    st.session_state["query_input"] = text[: settings.max_query_chars]


def render_gaps_and_contradictions(state: dict) -> None:
    """Coverage warnings (system-detected), follow-up research (model-suggested), contradictions."""
    items = state.get("gap_items") or []
    curated: dict = curate_gaps(items) if items else {"coverage": [], "open_questions": [], "dropped": 0}
    contras = str_list(state.get("contradictions"))
    legacy = [] if items else str_list(state.get("unresolved_gaps"))
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
            right.button("Research this", key=f"followup_{n}", on_click=start_followup, args=(o["text"],))

    if legacy:
        with st.expander(f"Gaps noted ({len(legacy)})"):
            for g in legacy:
                st.write(f"- {g}")

    if contras:
        with st.expander(f"⚡ Contradictions between sources ({len(contras)})", expanded=True):
            for c in contras:
                st.info(c)


def render_citations(citations: list) -> None:
    if not citations:
        return
    st.subheader(f"Sources consulted ({len(citations)})")
    seen: set[str] = set()
    for i, cit in enumerate(citations, 1):
        url     = attr(cit, "url",         "#") or "#"
        title   = attr(cit, "title",       "Untitled") or "Untitled"
        snippet = attr(cit, "snippet",     "")  or ""
        src     = attr(cit, "source_type", "web") or "web"
        if url in seen:
            continue
        seen.add(url)
        with st.expander(f"{i}. {title}  `[{src}]`"):
            if snippet:
                st.write(snippet)
            st.markdown(f"[{url}]({url})")


def render_related_past_reports(related: list) -> None:
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


def render_routing_history(state: dict) -> None:
    routing = str_list(state.get("routing_history"))
    if not routing:
        return
    with st.expander("Routing history (debug)", expanded=False):
        for entry in routing:
            st.code(entry, language=None)
