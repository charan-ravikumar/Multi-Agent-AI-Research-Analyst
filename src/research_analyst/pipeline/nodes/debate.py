"""Optional Critic <-> Synthesizer review loop (opt-in; see the README evaluation for why it is off by default)."""
from __future__ import annotations

from research_analyst.config import settings
from research_analyst.models import ResearchState
from research_analyst.observability.logger import get_logger
from research_analyst.observability.progress import progress
from research_analyst.observability.usage import BudgetExceeded
from research_analyst.pipeline.registry import critic, synthesizer

_debate_log = get_logger("debate")


def budget_stop(routing_entries: list[str], current_draft: list[str], session_id: str, round_num: int, reason: str) -> dict:
    routing_entries.append(f"debate | round={round_num + 1} | BUDGET_EXHAUSTED ({reason})")
    _debate_log.warning(f"debate stopped early: {reason}", step="debate_budget_exhausted", session_id=session_id)
    return {
        "critic_objections": [],
        "debate_round": round_num,
        "debate_resolved": True,
        "debate_forced": False,
        "debate_budget_exhausted": True,
        "revised_draft": current_draft,
        "routing_history": routing_entries,
    }


async def debate_loop_node(state: ResearchState) -> dict:
    """Progress-aware wrapper: whatever way the loop exits, the review stage is complete."""
    sid = state.get("session_id", "")
    try:
        return await debate_loop_node_inner(state)
    finally:
        progress.finish_stage(sid, "review", "Writing the final report…")


async def debate_loop_node_inner(state: ResearchState) -> dict:
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
    # per-run override (UI opt-in) wins over the global default
    max_rounds: int = int(state.get("review_rounds", settings.debate_max_rounds))
    current_draft: list[str] = list(state.get("draft", []))
    routing_entries: list[str] = []
    session_id_d: str = state.get("session_id", "")

    if max_rounds <= 0:      # review disabled (e.g. ablation baseline): pass the draft straight through
        return {
            "critic_objections": [],
            "debate_round": 0,
            "debate_resolved": True,
            "debate_forced": False,
            "revised_draft": current_draft,
            "routing_history": ["debate | SKIPPED (review loop off)"],
        }

    prev_objections: list[str] = []
    minor_notes: list[str] = []
    for round_num in range(max_rounds):
        # ── 1. Critique current draft ─────────────────────────────────────────────
        critic_state = {
            **state,
            "draft": current_draft,
            "debate_round": round_num,
            "critic_objections": [],   # clear stale objections from prior round
            "previous_objections": prev_objections,
        }
        progress.set_label(session_id_d, f"Reviewing the draft (round {round_num + 1})…")
        try:
            critic_out = await critic(critic_state)
        except BudgetExceeded as exc:
            return budget_stop(routing_entries, current_draft, session_id_d, round_num, str(exc))
        objections: list[str] = critic_out.get("critic_objections", [])
        blocking: int = critic_out.get("critic_blocking", len(objections))
        minor_notes = list(critic_out.get("critic_minor", []))

        routing_entries.append(
            f"debate | round={round_num + 1}/{max_rounds}"
            f" | objections={len(objections)} high={blocking} minor={len(minor_notes)}"
        )
        _debate_log.info(
            f"round {round_num + 1}/{max_rounds}: {len(objections)} objection(s)",
            step="debate_round",
            session_id=session_id_d,
        )

        progress.advance(session_id_d, "review", 1)
        # ── 1b. Critic output unparseable: the draft was NOT reviewed ────────────
        if critic_out.get("critic_review_skipped"):
            routing_entries.append(f"debate | round={round_num + 1} | REVIEW_SKIPPED (critic output unparseable)")
            _debate_log.warning(
                "critic output unparseable; review skipped (not approved)",
                step="debate_review_skipped",
                session_id=session_id_d,
            )
            return {
                "critic_objections": [],
                "debate_round": round_num,
                "debate_resolved": True,
                "debate_forced": False,
                "debate_review_skipped": True,
                "revised_draft": current_draft,
                "routing_history": routing_entries,
            }

        # ── 2. Organic resolution: nothing HIGH-severity left (low/medium may remain) ──
        if blocking == 0:
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
                "review_notes": objections + minor_notes,   # unaddressed medium/low points, disclosed
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
        prev_objections = list(objections)
        revision_state = {
            **state,
            "draft": current_draft,
            "critic_objections": objections,
        }
        progress.set_label(session_id_d, f"Revising the draft (round {round_num + 1})…")
        try:
            revision_out = await synthesizer(revision_state)
        except BudgetExceeded as exc:
            return budget_stop(routing_entries, current_draft, session_id_d, round_num, str(exc))
        revised = revision_out.get("revised_draft", [])
        if revised:
            current_draft = revised
        progress.advance(session_id_d, "review", 1)
        if revision_out.get("revision_failed"):
            routing_entries.append(
                f"debate | round={round_num + 1} | REVISION_FAILED (kept previous draft)"
                f" | sections={len(current_draft)}"
            )
        else:
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
