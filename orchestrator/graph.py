"""
orchestrator/graph.py â€” LangGraph StateGraph for the research pipeline

Topology (Stage 2 â€” parallel fan-out per sub-question):

    START
      -> planner            PlannerAgent         query -> sub_questions
      -> [conditional]      _fan_out             returns [Send("process_sub_question", ...)]
                                                 one Send per sub-question (parallel)
      -> process_sub_question  _process_sub_question  composite node:
                                 Searcher -> [optional retry] -> Reader -> Synthesizer
                                 accumulates to: search_results, extracted_facts,
                                                 draft, citations, contradictions,
                                                 unresolved_gaps, routing_history
      -> supervisor         _supervisor_node     logs fan-in summary -> writer (unconditional)
      -> writer             WriterAgent          -> END

Fan-out mechanics (verified in tests/test_send_api.py):
  - LangGraph Send() passes the Send arg as the input state to the target node
  - Chained add_edge() nodes after a Send target run on GLOBAL state (not branch-local)
  - Therefore the full Searcher->Reader->Synthesizer pipeline is ONE composite function,
    not three graph nodes connected by edges
  - Each branch returns only incremental outputs; operator.add reducers merge them

Zero-results retry:
  - Handled INSIDE _process_sub_question (one retry per branch, capped at 1).
  - There is NO graph-level retry path.  The old supervisor->searcher conditional edge
    was removed (deliberate architectural change): current_sub_question is branch-local
    (set only in Send() args, never written back to global state), so the supervisor
    always sees current_sub_question="" and any go_searcher condition is permanently
    False.  Verified in tests/test_v2_retry_parallel.py.

LangGraph version: 1.2.7
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from agents.planner import PlannerAgent
from agents.critic import CriticAgent
from agents.reader import ReaderAgent
from agents.searcher import SearcherAgent
from agents.synthesizer import SynthesizerAgent
from agents.writer import WriterAgent
from config import settings
from core.logger import get_logger
from models import ResearchState

# â”€â”€ agent singletons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_planner = PlannerAgent()
_searcher = SearcherAgent()
_reader = ReaderAgent()
_synthesizer = SynthesizerAgent()
_critic = CriticAgent()
_writer = WriterAgent()

_branch_log = get_logger("branch")
_debate_log  = get_logger("debate")
_plan_log    = get_logger("plan")
_memory_log  = get_logger("memory")


# â”€â”€ fan-out function (used as conditional edge path after planner) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fan_out(state: ResearchState) -> list:
    """
    Called by add_conditional_edges after the Planner node.
    Returns one Send per sub-question; LangGraph dispatches them in parallel.
    Each Send passes the full current state with current_sub_question overridden
    so _process_sub_question knows which sub-question to handle.
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        raise ValueError("Planner produced no sub_questions â€” cannot fan out.")
    return [
        Send("process_sub_question", {**state, "current_sub_question": sq})
        for sq in sub_questions
    ]

# -- human-in-the-loop plan approval node ------------------------------------

async def _plan_approval_node(state: ResearchState) -> dict:
    """
    Human checkpoint between Planner output and fan-out.

    Also performs a best-effort cross-run memory lookup (keyword matching,
    NOT semantic search — see memory/redis_store.find_related_reports) and
    populates state["related_past_reports"] with any topically overlapping
    past reports found in Redis.

    Behaviour:
      - auto_approve_plan=True  (automated tests / CI): returns immediately,
        sub_questions unchanged.  related_past_reports still populated.
      - auto_approve_plan absent or False: calls interrupt() to surface the
        sub_questions (and any related past work) to the caller.  Execution
        pauses until the graph is resumed via Command(resume=<approved_list>).
        The resume value replaces state["sub_questions"] so the fan-out
        dispatches the approved set.

    Verified mechanism (tests/_verify_interrupt.py):
      interrupt(value) pauses the graph and returns the value in
      result["__interrupt__"].  Command(resume=v) resumes; interrupt()
      returns v inside the node.  The node re-runs from the top on resume
      but interrupt() returns immediately with the stored value.
      Requires graph compiled with a checkpointer AND thread_id in config.
    """
    sub_questions: list[str] = list(state.get("sub_questions", []))

    # ── cross-run memory lookup ──────────────────────────────────────────────
    # Best-effort: if Redis is unavailable the pipeline continues without
    # related-report context (graceful degradation).
    related: list[dict] = []
    try:
        from memory.redis_store import find_related_reports
        related = await asyncio.to_thread(
            find_related_reports, state.get("query", "")
        )
        if related:
            _plan_log.info(
                f"Found {len(related)} related past report(s) for query={state.get('query', '')[:60]!r}",
                step="memory_lookup",
                session_id=state.get("session_id", ""),
            )
    except Exception as exc:
        _plan_log.warning(
            f"Related-report lookup failed (Redis unavailable?): {exc}",
            step="memory_lookup_fail",
            session_id=state.get("session_id", ""),
        )

    if state.get("auto_approve_plan", False):
        # Automated / test path: skip human review.
        return {
            "related_past_reports": related,
            "routing_history": [
                f"plan | auto_approved"
                f" | sub_questions={len(sub_questions)}"
                f" | related_found={len(related)}"
            ],
        }

    # Human-in-the-loop path: surface plan + related past work for review.
    # NOTE: Because LangGraph re-runs the node from the top on resume,
    # this block executes twice: once before the pause and once on resume
    # (where interrupt() immediately returns the stored value).
    print("\n" + "=" * 62)
    print("HUMAN CHECKPOINT \u2014 Research Plan Approval")
    print("=" * 62)
    print(f"Query: {state.get('query', '')}")
    print(f"\nProposed sub-questions ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    if related:
        print(f"\nRelated past reports ({len(related)}):")
        for r in related[:3]:
            print(f"  [{r.get('timestamp_human','?')}] {r.get('query','')[:80]}")
            print(f"    Shared tokens: {r.get('_overlap_tokens', [])}")
    print("\nResume with: Command(resume=<approved_list>)")
    print("  Approve as-is : Command(resume=state['sub_questions'])")
    print("  Edit          : Command(resume=[sq1, sq2, ...])")
    print("=" * 62 + "\n")

    approved: list[str] = interrupt({
        "checkpoint": "plan_approval",
        "query": state.get("query", ""),
        "sub_questions": sub_questions,
        "related_past_reports": related,
        "message": (
            "Review the proposed sub-questions. "
            "Resume with the approved list (edit as needed)."
        ),
    })

    return {
        "sub_questions": approved,
        "related_past_reports": related,
        "routing_history": [
            f"plan | human_approved"
            f" | approved={len(approved)}/{len(sub_questions)}"
            f" | related_found={len(related)}"
        ],
    }

# â”€â”€ composite branch node â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def _process_sub_question(state: dict) -> dict:
    """
    Composite branch node: runs the full Searcher -> Reader -> Synthesizer
    pipeline for ONE sub-question in a single Python function.

    Must be a single function (not chained graph nodes) because LangGraph's
    Send() only preserves the Send arg (current_sub_question) for the FIRST
    node; subsequent nodes connected via add_edge run on global state where
    current_sub_question is lost. See tests/test_send_api.py for the verified
    proof.

    Handles zero-results retry internally (capped at 1 attempt).
    Returns ONLY incremental outputs â€” LangGraph's operator.add reducers
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

    # â”€â”€ 1. Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    search_out = await _searcher(state)
    new_results = search_out.get("search_results", [])
    new_failed  = search_out.get("failed_sub_questions", [])

    all_new_results = list(new_results)
    all_new_failed  = list(new_failed)

    # â”€â”€ 2. Zero-results retry (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if sq in new_failed:
        routing_entries.append(
            f"process_sub_question | [{sq}] retry_searcher | zero_results=True"
        )
        retry_out    = await _searcher(state)
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

    # â”€â”€ 3. Read â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Build reader input: original Send arg state + this branch's search results.
    # Reader filters search_results by current_sub_question, so including global
    # state's results for other sub_questions is harmless (they get filtered out).
    reader_input = {
        **state,
        "search_results":      state.get("search_results", []) + all_new_results,
        "failed_sub_questions": state.get("failed_sub_questions", []) + all_new_failed,
    }
    reader_out = await _reader(reader_input)
    new_facts  = reader_out.get("extracted_facts", [])

    # â”€â”€ 4. Synthesize â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Build synth input: reader_input + this branch's extracted facts.
    synth_input = {
        **reader_input,
        "extracted_facts": reader_input.get("extracted_facts", []) + new_facts,
    }
    synth_out = await _synthesizer(synth_input)

    elapsed = time.perf_counter() - t0
    ts_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _branch_log.info(
        f"BRANCH_END [{sq}] at {ts_end} | elapsed={elapsed:.1f}s",
        step="branch_end",
        session_id=session_id_br,
        latency_s=round(elapsed, 3),
    )

    # â”€â”€ Return incremental outputs only (operator.add reducers merge them) â”€â”€â”€â”€
    n_results  = len(all_new_results)
    n_facts    = len(new_facts)
    n_sections = len(synth_out.get("draft", []))
    routing_entries.append(
        f"branch | [{sq}] results={n_results} facts={n_facts} sections={n_sections}"
    )
    result: dict = {
        "search_results":       all_new_results,
        "failed_sub_questions": all_new_failed,
        "extracted_facts":      new_facts,
        "citations":            synth_out.get("citations", []),
        "contradictions":       synth_out.get("contradictions", []),
        "unresolved_gaps":      synth_out.get("unresolved_gaps", []),
        "draft":                synth_out.get("draft", []),
        "routing_history":      routing_entries,
    }
    return result


# -- supervisor node ---------------------------------------------------

def _supervisor_node(state: ResearchState) -> dict:
    """
    Fan-in observer -- runs after all parallel branches complete.

    Records a summary entry in routing_history, then the graph proceeds
    unconditionally to the Writer via a plain add_edge().

    There is no graph-level retry path.  Zero-results retries are handled
    inside _process_sub_question (branch-local, capped at 1 attempt).
    A supervisor->searcher conditional edge was removed (deliberate
    architectural change): current_sub_question is set only in Send() args
    (branch-local) and never written back to global state, so any go_searcher
    condition would be permanently False.  See module docstring and
    tests/test_v2_retry_parallel.py.
    """
    n_sq    = len(state.get("sub_questions", []))
    n_facts = len(state.get("extracted_facts", []))
    n_gaps  = len(state.get("unresolved_gaps", []))

    log_entry = (
        f"supervisor | {n_sq} branch(es) complete -> debate"
        f" | facts={n_facts} gaps={n_gaps}"
    )
    return {"routing_history": [log_entry]}


# -- debate loop node ---------------------------------------------------------

async def _debate_loop_node(state: ResearchState) -> dict:
    """
    Bounded Critic <-> Synthesizer debate loop (max settings.debate_max_rounds).

    Runs after the supervisor fan-in on the FULLY assembled state:
      state["draft"]            -- List[str] from all parallel branches
      state["contradictions"]   -- merged across all branches
      state["unresolved_gaps"]  -- merged across all branches

    This is the right place to catch cross-sub-question contradictions because
    all branches have already merged via operator.add.  A per-branch critic
    would only see its own data.

    Each round:
      1. CriticAgent reviews assembled draft -> produces critic_objections
      2. If no objections -> organic resolution, exit.
      3. If round cap hit -> forced resolution, exit.
      4. Otherwise SynthesizerAgent (revision mode) revises the draft.
         Revision mode is detected by critic_objections being non-empty.

    The draft field uses operator.add (fan-in reducer), so it cannot be
    safely overwritten post-fan-in.  Revised text is returned in
    revised_draft (plain field, last-write-wins).  Writer reads
    revised_draft preferentially.
    """
    max_rounds: int = settings.debate_max_rounds
    current_draft: list[str] = list(state.get("draft", []))
    routing_entries: list[str] = []
    session_id_d: str = state.get("session_id", "")

    for round_num in range(max_rounds):
        # ── 1. Critique current draft ─────────────────────────────────────────────
        critic_state = {
            **state,
            "draft": current_draft,
            "debate_round": round_num,
            "critic_objections": [],   # clear stale objections from prior round
        }
        critic_out = await _critic(critic_state)
        objections: list[str] = critic_out.get("critic_objections", [])

        routing_entries.append(
            f"debate | round={round_num + 1}/{max_rounds}"
            f" | objections={len(objections)}"
        )
        _debate_log.info(
            f"round {round_num + 1}/{max_rounds}: {len(objections)} objection(s)",
            step="debate_round",
            session_id=session_id_d,
        )

        # ── 2. Organic resolution ───────────────────────────────────────────────
        if not objections:
            routing_entries.append(
                f"debate | resolved=organic | round={round_num + 1}"
            )
            _debate_log.info(
                f"organically resolved at round {round_num + 1}",
                step="debate_resolved",
                session_id=session_id_d,
            )
            return {
                "critic_objections": [],
                "debate_round": round_num + 1,
                "debate_resolved": True,
                "debate_forced": False,
                "revised_draft": current_draft,
                "routing_history": routing_entries,
            }

        # ── 3. Forced resolution (cap) ──────────────────────────────────────────
        if round_num == max_rounds - 1:
            routing_entries.append(
                f"debate | resolved=FORCED | cap={max_rounds}"
                f" | unresolved={len(objections)}"
            )
            _debate_log.warning(
                f"cap hit at round {max_rounds};"
                f" {len(objections)} objection(s) unresolved",
                step="debate_forced",
                session_id=session_id_d,
            )
            return {
                "critic_objections": objections,
                "debate_round": max_rounds,
                "debate_resolved": True,
                "debate_forced": True,
                "revised_draft": current_draft,
                "routing_history": routing_entries,
            }

        # ── 4. Revise: Synthesizer in revision mode ─────────────────────────────
        # Revision mode is triggered by critic_objections being non-empty.
        # The Synthesizer returns {"revised_draft": [section, ...]}
        revision_state = {
            **state,
            "draft": current_draft,
            "critic_objections": objections,
        }
        revision_out = await _synthesizer(revision_state)
        revised = revision_out.get("revised_draft", [])
        if revised:
            current_draft = revised
        routing_entries.append(
            f"debate | round={round_num + 1} | revised | sections={len(current_draft)}"
        )

    # Defensive fallback (loop should never exhaust without returning above)
    return {
        "critic_objections": [],
        "debate_round": max_rounds,
        "debate_resolved": True,
        "debate_forced": True,
        "revised_draft": current_draft,
        "routing_history": routing_entries,
    }


# â”€â”€ graph construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# ── cross-run memory store node ───────────────────────────────────────────────

async def _memory_node(state: ResearchState) -> dict:
    """
    Persist a compact record of the completed report to Redis for cross-run
    memory.  Runs as the final graph node (after Writer).

    Best-effort: a Redis failure is logged as a WARNING but does NOT crash the
    pipeline.  The final_report is still fully available in state.

    Stored fields per report: report_id, session_id, query, sub_questions,
    executive_summary (first 500 chars), timestamp, timestamp_human.
    """
    try:
        from memory.redis_store import save_report_record
        report_id = await asyncio.to_thread(
            save_report_record,
            session_id=state.get("session_id", "unknown"),
            query=state.get("query", ""),
            final_report=state.get("final_report", ""),
            sub_questions=list(state.get("sub_questions", [])),
        )
        _memory_log.info(
            f"Report persisted to Redis: report_id={report_id[:8]}",
            step="memory_write",
            session_id=state.get("session_id", ""),
        )
        return {"routing_history": [f"memory | report saved | id={report_id[:8]}"]}
    except Exception as exc:
        _memory_log.warning(
            f"Memory write failed (Redis unavailable?): {exc}",
            step="memory_write_fail",
            session_id=state.get("session_id", ""),
        )
        return {}

def build_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the research pipeline.

    Args:
        checkpointer: LangGraph checkpointer for persistence and interrupt
            support.  Pass MemorySaver() (or any other checkpointer) to enable
            interrupt()-based human-in-the-loop checkpoints.  Callers MUST
            then include config={"configurable": {"thread_id": ...}} in every
            invoke() / ainvoke() call.  Pass None to compile without a
            checkpointer (interrupt() will not work but no thread_id needed).
    """
    graph = StateGraph(ResearchState)

    # nodes
    graph.add_node("planner",              _planner)
    graph.add_node("plan_approval",        _plan_approval_node)
    graph.add_node("process_sub_question", _process_sub_question)
    graph.add_node("supervisor",           _supervisor_node)
    graph.add_node("debate",               _debate_loop_node)
    graph.add_node("writer",               _writer)
    graph.add_node("memory_store",          _memory_node)

    # edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "plan_approval")               # -> human checkpoint
    graph.add_conditional_edges("plan_approval", _fan_out)   # -> fan-out
    graph.add_edge("process_sub_question", "supervisor")     # fan-in
    graph.add_edge("supervisor", "debate")                   # -> debate loop
    graph.add_edge("debate", "writer")                       # -> writer
    graph.add_edge("writer", "memory_store")         # -> persist report
    graph.add_edge("memory_store", END)

    return graph.compile(checkpointer=checkpointer)


# â”€â”€ module-level compiled app (import-ready) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Compiled with MemorySaver so interrupt() works out of the box.
# ALL callers must pass config={"configurable": {"thread_id": ...}} to
# every invoke() / ainvoke() call.  Automated pipelines should also set
# auto_approve_plan=True in initial state to bypass the human checkpoint.
app = build_graph(checkpointer=MemorySaver())
