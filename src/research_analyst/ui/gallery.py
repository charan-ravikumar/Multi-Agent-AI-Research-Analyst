"""
Sample-report gallery: pre-generated reports shown without spending any API quota.

Each entry is a JSON file in ui/gallery/ produced by ``scripts/generate_gallery.py`` from a REAL pipeline
run (same code path as a live query).  An entry stores the report, its references, grounding stats, the
curated coverage warnings / follow-up questions, run cost, and the evidence needed by the click-to-verify
panel — so a visitor can inspect quality, and verify claims, without a single LLM call.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from research_analyst.grounding.gaps import curate_gaps
from research_analyst.grounding.verify import verify_payload
from research_analyst.ui import components as C

GALLERY_DIR = Path(__file__).parent / "gallery"
_EVIDENCE_CHARS = 900          # per source: enough to verify a claim, small enough to keep entries compact


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def load_gallery(directory: Path | None = None) -> list[dict]:
    """All entries, sorted by question.  Malformed files are skipped, never fatal."""
    entries = []
    for p in sorted((directory or GALLERY_DIR).glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
            if e.get("question") and e.get("report"):
                entries.append(e)
        except (OSError, ValueError):
            continue
    return entries


def entry_from_state(state: dict, question: str, seconds: float, models: dict[str, str], review_loop: bool = False) -> dict[str, Any]:
    """Build a gallery entry from a finished run's state."""
    curated = curate_gaps(state.get("gap_items") or [])
    verify = verify_payload(state["final_report"], state.get("search_results", []), state.get("extracted_facts", []),
                            state.get("references", []), state.get("claim_verdicts"))
    for ev in verify["evidence"].values():
        ev["text"] = ev["text"][:_EVIDENCE_CHARS]
    usage = state.get("usage_summary") or {}
    return {
        "id": slugify(question),
        "question": question,
        "generated_at": date.today().isoformat(),
        "models": models,
        "review_loop": review_loop,
        "seconds": round(seconds, 1),
        "usage": {"total_tokens": usage.get("total_tokens"), "calls": usage.get("calls"),
                  "estimated_cost_usd": usage.get("estimated_cost_usd"), "by_agent": usage.get("by_agent"),
                  "by_model": usage.get("by_model"), "truncated_calls": usage.get("truncated_calls")},
        "report": state["final_report"],
        "references": state.get("references", []),
        "citation_stats": state.get("citation_stats", {}),
        "coverage": [c["text"] for c in curated["coverage"]],
        "open_questions": [{"text": o["text"], "sub_question": o["sub_question"]} for o in curated["open_questions"]],
        "contradictions": list(state.get("contradictions", [])),
        "verify": verify,
    }


def render_gallery(directory: Path | None = None) -> None:
    entries = load_gallery(directory)
    st.markdown("### Sample reports")
    st.caption("Real outputs of this pipeline, generated in advance — viewing them uses no API quota. "
               "Each shows the report, how well it is grounded, what it cost, and lets you verify claims against their sources.")
    if not entries:
        st.info("No sample reports are bundled with this build yet.")
        return
    labels = [e["question"] for e in entries]
    choice = st.selectbox("Choose a report", labels, key="gallery_choice")
    e = entries[labels.index(choice)]
    models = e.get("models", {})
    st.caption(f"Generated {e.get('generated_at', '?')} with {models.get('strong', '?')} · review loop "
               f"{'on' if e.get('review_loop') else 'off'} · AI-generated: check the sources before relying on it.")
    st.markdown(e["report"])
    C.render_citation_stats({"citation_stats": e.get("citation_stats", {})})
    st.divider()
    C.render_run_summary({"calls": (e.get("usage") or {}).get("calls"), "total_tokens": (e.get("usage") or {}).get("total_tokens") or 0,
                          "estimated_cost_usd": (e.get("usage") or {}).get("estimated_cost_usd"),
                          "by_agent": (e.get("usage") or {}).get("by_agent"), "by_model": (e.get("usage") or {}).get("by_model"),
                          "truncated_calls": (e.get("usage") or {}).get("truncated_calls")},
                         e.get("seconds"), key="gallery")
    v = e.get("verify") or {}
    C.render_verify(v.get("claims", []), v.get("evidence", {}), key=f"gallery_{e['id']}", verdicts=v.get("verdicts"))
    if e.get("coverage"):
        st.subheader("⚠ Coverage warnings")
        for c in e["coverage"]:
            st.warning(c)
    if e.get("open_questions"):
        st.subheader("🔎 Suggested follow-up research")
        for o in e["open_questions"]:
            st.markdown(f"- **{o['text']}**  \n  <span style='opacity:.6'>from: {o['sub_question'][:90]}</span>", unsafe_allow_html=True)
