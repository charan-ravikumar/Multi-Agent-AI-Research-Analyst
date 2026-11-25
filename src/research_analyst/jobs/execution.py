"""
jobs/execution.py — run ONE stage of a run from wherever its checkpoint says it is.

The same call handles a first attempt and a redelivery after a crash: the checkpoint, not the job, decides what is
left to do.  Sub-question branches that finished before a crash are stored as pending writes, so LangGraph does not
run them again; a branch that was in flight starts over.
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Command

from research_analyst.jobs import store as S
from research_analyst.models.research_state import new_research_state
from research_analyst.observability.progress import progress

_BRANCH_NODE = "process_sub_question"


class RunStateMissing(RuntimeError):
    """The checkpoint of an approved run no longer exists (expired or flushed)."""


def run_config(run_id: str, review_rounds: int = 0) -> dict:
    return {"configurable": {"thread_id": run_id, "session_id": run_id},
            "tags": ["review:on" if review_rounds else "review:off", "execution:queue"]}


async def execute_stage(graph: Any, stage: str, record: dict[str, Any]) -> tuple[str, dict | None]:
    """Returns (status, state): ``awaiting_approval`` after planning, ``done`` with the final state after research."""
    run_id = record["run_id"]
    rounds = int(record.get("review_rounds") or 0)
    cfg = run_config(run_id, rounds)
    snapshot = await graph.aget_state(cfg)

    if stage == S.PLAN:
        if snapshot.interrupts:                       # planned, but the worker died before reporting it
            return S.AWAITING_APPROVAL, None
        first = new_research_state(run_id, record["query"], review_rounds=rounds)
        # ``durability="sync"``: a step's checkpoint is stored before the next one starts.
        result = await graph.ainvoke(None if snapshot.values else first, config=cfg, durability="sync")
        return (S.AWAITING_APPROVAL, None) if result.get("__interrupt__") else (S.DONE, result)

    if not snapshot.values:
        raise RunStateMissing("The saved research plan has expired. Start the research again.")
    if not snapshot.interrupts and not snapshot.next:
        return S.DONE, snapshot.values                # finished, but the worker died before reporting it
    approved = list(record.get("approved") or [])
    _restore_progress(run_id, snapshot, len(approved), rounds)
    command: Command | None = Command(resume=approved) if snapshot.interrupts else None
    return S.DONE, await graph.ainvoke(command, config=cfg, durability="sync")


def _restore_progress(run_id: str, snapshot: Any, n_branches: int, review_rounds: int) -> None:
    """Start the progress bar, counting the branches an earlier attempt already completed."""
    progress.start(run_id, n_branches, review_rounds)
    if snapshot.interrupts or "plan_approval" in snapshot.next:
        return
    branches = [t for t in snapshot.tasks if t.name == _BRANCH_NODE]
    finished = sum(1 for t in branches if t.result is not None) if branches else n_branches
    for _ in range(finished):
        progress.branch(run_id).close("Resumed: reusing finished research")
