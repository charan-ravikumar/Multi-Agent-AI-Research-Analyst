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


def render_usage(state: dict) -> None:
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
