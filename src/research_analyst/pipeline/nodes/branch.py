"""Per-sub-question branch: fan-out, and the failure-isolated Searcher -> Reader -> Synthesizer node.

Why ONE composite node (not three graph nodes): LangGraph's Send() preserves its argument only for
the first target node; chained add_edge() nodes run on GLOBAL state where current_sub_question is
lost (proved in tests/manual/test_send_api.py).  Each branch returns only incremental outputs;
operator.add reducers merge them.  Zero-results retry (once) lives inside the branch.
"""
from __future__ import annotations

from typing import Any

import time
from datetime import datetime

from langgraph.types import Send

from research_analyst.grounding.gaps import COVERAGE, make_item
from research_analyst.models import ResearchState
from research_analyst.observability.logger import get_logger
from research_analyst.observability.progress import progress
from research_analyst.pipeline.registry import reader, searcher, synthesizer

_branch_log = get_logger("branch")


def fan_out(state: ResearchState) -> list:
    """
    Called by add_conditional_edges after the Planner node.
    Returns one Send per sub-question; LangGraph dispatches them in parallel.
    Each Send passes the full current state with current_sub_question overridden
    so process_sub_question knows which sub-question to handle.
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        raise ValueError("Planner produced no sub_questions — cannot fan out.")
    return [
        Send("process_sub_question", {**state, "current_sub_question": sq})
        for sq in sub_questions
    ]


async def process_sub_question(state: dict) -> dict:
    """
    Failure-isolating wrapper around the branch pipeline.

    One sub-question failing (provider outage, unexpected exception) must not
    discard the other branches' completed work.  On any Exception the branch
    returns a recorded gap + routing entry instead of raising; the run only
    aborts if EVERY branch failed (see supervisor_node).

    The sub-question is stripped here once so every downstream equality check
    (search results, facts, failed list) compares identical strings.
    """
    sq = state.get("current_sub_question", "").strip()
    state = {**state, "current_sub_question": sq}
    br = progress.branch(state.get("session_id", ""))
    try:
        return await process_sub_question_inner(state, br)
    except Exception as exc:  # noqa: BLE001 — isolate the branch; CancelledError is BaseException
        _branch_log.error(
            f"BRANCH_FAILED [{sq}]: {type(exc).__name__}: {exc}",
            step="branch_failed",
            session_id=state.get("session_id", ""),
        )
        msg = (f'Research for sub-question "{sq}" failed and was skipped '
               f"({type(exc).__name__}: {str(exc)[:160]})")
        return {
            "failed_sub_questions": [sq],
            "unresolved_gaps": [msg],
            "gap_items": [make_item(sq, COVERAGE, msg)],
            "routing_history": [f"branch_failed | [{sq}] {type(exc).__name__}: {str(exc)[:160]}"],
        }
    finally:
        br.close()


async def process_sub_question_inner(state: dict, br: Any = None) -> dict:
    """
    Composite branch node: runs the full Searcher -> Reader -> Synthesizer
    pipeline for ONE sub-question in a single Python function.

    Must be a single function (not chained graph nodes) because LangGraph's
    Send() only preserves the Send arg (current_sub_question) for the FIRST
    node; subsequent nodes connected via add_edge run on global state where
    current_sub_question is lost. See tests/test_send_api.py for the verified
    proof.

    Handles zero-results retry internally (capped at 1 attempt).
    Returns ONLY incremental outputs — LangGraph's operator.add reducers
    accumulate them into global state as branches fan in.
    """
    sq: str = state.get("current_sub_question", "")
    session_id_br: str = state.get("session_id", "")
    ts_start = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    t0 = time.perf_counter()
    _branch_log.info(
        f"BRANCH_START [{sq}] at {ts_start}",
        step="branch_start",
        session_id=session_id_br,
    )

    routing_entries: list = []

    # ── 1. Search ─────────────────────────────────────────────────────────────
    search_out = await searcher(state)
    if br:
        br.step(f"Searched sources for: {sq[:70]}")
    new_results = search_out.get("search_results", [])
    new_failed  = search_out.get("failed_sub_questions", [])

    all_new_results = list(new_results)
    all_new_failed  = list(new_failed)

    # ── 2. Zero-results retry (once) ─────────────────────────────────────────
    if sq in new_failed:
        routing_entries.append(
            f"process_sub_question | [{sq}] retry_searcher | zero_results=True"
        )
        retry_out    = await searcher(state)
        retry_results = retry_out.get("search_results", [])
        all_new_results += retry_results
        if retry_results:
            all_new_failed = [f for f in all_new_failed if f != sq]
            routing_entries.append(
                f"process_sub_question | [{sq}] retry succeeded"
                f" | results={len(retry_results)}"
            )
        else:
            routing_entries.append(
                f"process_sub_question | [{sq}] retry failed | still zero results"
            )

    # ── 3. Read ───────────────────────────────────────────────────────────────
    # Build reader input: original Send arg state + this branch's search results.
    # Reader filters search_results by current_sub_question, so including global
    # state's results for other sub_questions is harmless (they get filtered out).
    reader_input = {
        **state,
        "search_results":      state.get("search_results", []) + all_new_results,
        "failed_sub_questions": state.get("failed_sub_questions", []) + all_new_failed,
    }
    reader_out = await reader(reader_input)
    new_facts  = reader_out.get("extracted_facts", [])
    if br:
        br.step(f"Read sources for: {sq[:70]}")

    # ── 4. Synthesize ─────────────────────────────────────────────────────────
    # Build synth input: reader_input + this branch's extracted facts.
    synth_input = {
        **reader_input,
        "extracted_facts": reader_input.get("extracted_facts", []) + new_facts,
    }
    synth_out = await synthesizer(synth_input)
    if br:
        br.step(f"Wrote section: {sq[:70]}")

    elapsed = time.perf_counter() - t0
    ts_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _branch_log.info(
        f"BRANCH_END [{sq}] at {ts_end} | elapsed={elapsed:.1f}s",
        step="branch_end",
        session_id=session_id_br,
        latency_s=round(elapsed, 3),
    )

    # ── Return incremental outputs only (operator.add reducers merge them) ────
    rs = reader_out.get("reader_stats") or {}
    read_note = (f" read={rs['sources_read']}/{rs.get('sources_available', '?')}src in {rs['batches']} call(s)"
                 f" stop={rs['stopped']}") if rs else ""
    n_results  = len(all_new_results)
    n_facts    = len(new_facts)
    n_sections = len(synth_out.get("draft", []))
    routing_entries.append(
        f"branch | [{sq}] results={n_results} facts={n_facts} sections={n_sections}{read_note}"
    )
    result: dict = {
        "search_results":       all_new_results,
        "failed_sub_questions": all_new_failed,
        "extracted_facts":      new_facts,
        "citations":            synth_out.get("citations", []),
        "contradictions":       synth_out.get("contradictions", []),
        "unresolved_gaps":      synth_out.get("unresolved_gaps", []),
        "gap_items":            synth_out.get("gap_items", []),
        "draft":                synth_out.get("draft", []),
        "routing_history":      routing_entries,
    }
    return result
