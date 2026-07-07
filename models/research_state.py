"""
models/research_state.py — LangGraph state definition

ResearchState is the ONLY contract between agents.
Every agent receives ResearchState → does its work → returns ResearchState.
No exceptions.

Node ↔ field ownership map (read → write):
  Planner      query            → research_plan, sub_questions
  Searcher     sub_questions, current_sub_question → search_results
  Reader       search_results   → extracted_facts
  Synthesizer  extracted_facts  → draft, citations, contradictions, unresolved_gaps
               (revision mode)  critic_objections, draft → revised_draft
  Critic       draft, contradictions, unresolved_gaps → critic_objections
  Debate loop  (orchestrates Critic + Synthesizer rounds) → critic_objections,
               debate_round, debate_resolved, debate_forced, revised_draft
  Writer       revised_draft | draft, critique  → final_report

The Annotated[List[...], operator.add] reducer lets parallel Searcher branches
merge their results automatically without extra orchestrator logic.
"""
from __future__ import annotations

import operator
from typing import Annotated, List, Optional

from typing_extensions import NotRequired, TypedDict

from models.citation import Citation
from models.critique import Critique
from models.fact import Fact
from models.research_plan import ResearchPlan
from models.search_result import SearchResult


class ResearchState(TypedDict):
    # ── session identity ──────────────────────────────────────────────────────
    session_id: str
    query: str

    # ── planner output ────────────────────────────────────────────────────────
    research_plan: Optional[ResearchPlan]   # full plan + metadata
    sub_questions: List[str]                # promoted to top-level so the
                                            # Searcher fans out without unpacking

    # ── searcher inbox ────────────────────────────────────────────────────────
    # Set by the orchestrator before dispatching each parallel Searcher branch.
    # NotRequired so existing construction sites (tests, planner) don't need it.
    current_sub_question: NotRequired[str]

    # ── searcher output ───────────────────────────────────────────────────────
    search_results: Annotated[List[SearchResult], operator.add]
    # Sub-questions that returned zero search results. Synthesizer uses this
    # to note coverage gaps in the draft rather than silently skipping them.
    failed_sub_questions: Annotated[List[str], operator.add]

    # ── reader output ─────────────────────────────────────────────────────────
    extracted_facts: Annotated[List[Fact], operator.add]

    # ── synthesizer output ────────────────────────────────────────────────────
    citations: Annotated[List[Citation], operator.add]
    # Each parallel Synthesizer branch appends its own synthesis paragraph.
    # operator.add accumulates sections in arrival order; WriterAgent joins
    # them before passing to the LLM prompt.
    draft: Annotated[List[str], operator.add]
    # Contradicting claims between sources, flagged explicitly rather than
    # silently resolved. operator.add merges parallel branch outputs.
    contradictions: Annotated[List[str], operator.add]
    # Knowledge gaps: facts don't fully answer the sub-question, plus any
    # failed_sub_questions promoted here so Critic/Writer see them explicitly.
    unresolved_gaps: Annotated[List[str], operator.add]

    # ── critic output ─────────────────────────────────────────────────────────
    critique: Optional[Critique]

    # ── debate loop control ───────────────────────────────────────────────────
    # Specific, addressable objections produced by CriticAgent each round.
    # Plain list (no operator.add): debate loop is sequential, not parallel.
    # NotRequired — absent in initial state; first written by _debate_loop_node.
    critic_objections: NotRequired[List[str]]

    # Number of Critic-Synthesizer rounds completed.
    debate_round: NotRequired[int]

    # True when the debate loop exits (either organic resolution or cap hit).
    debate_resolved: NotRequired[bool]

    # True when debate_resolved was forced by the round cap rather than the
    # Critic running out of objections.  Writer uses this to disclose
    # unresolved issues to the reader.
    debate_forced: NotRequired[bool]

    # Debate-revised draft produced by the Synthesizer revision pass.
    # Writer reads this preferentially over state["draft"] (which uses
    # operator.add and cannot be safely overwritten post-fan-in).
    revised_draft: NotRequired[List[str]]

    # ── writer output ─────────────────────────────────────────────────────────
    final_report: str

    # ── orchestrator control ──────────────────────────────────────────────────
    iteration: int      # reflection-loop counter; orchestrator checks vs
                        # settings.max_reflection_iterations before re-routing

    # When True the _plan_approval_node skips interrupt() and auto-approves the
    # Planner's sub_questions without human review.  Set this in automated tests
    # and CI pipelines.  Leave absent or False for interactive / human-in-the-loop
    # use where the human checkpoint should fire.
    auto_approve_plan: NotRequired[bool]

    # Supervisor routing trace — every decision is appended so the full path is
    # observable after the run.  operator.add keeps entries from all branches.
    routing_history: Annotated[List[str], operator.add]

    # Past reports whose topics overlap with the current query, surfaced before
    # the human checkpoint so the researcher can see what has already been done.
    # Populated by _plan_approval_node via find_related_reports() (keyword
    # matching — see memory/redis_store.py for documented limitations).
    # NotRequired: absent until the plan-approval node runs; not accumulated
    # across branches (plain field, last-write-wins).
    related_past_reports: NotRequired[List[dict]]


