"""Post-writer node: LLM-judge every cited claim of the final report.  Never fails the run."""
from __future__ import annotations

from research_analyst.config import settings
from research_analyst.models import ResearchState
from research_analyst.observability.logger import get_logger
from research_analyst.observability.progress import progress
from research_analyst.observability.usage import tracker
from research_analyst.pipeline import registry

_log = get_logger("grounding_node")


async def grounding_judge_node(state: ResearchState) -> dict:
    """Runs the GroundingJudge when enabled; on any failure returns an empty verdict set (UI falls back to lexical)."""
    if not settings.grounding_judge or not state.get("final_report"):
        return {}
    sid = state.get("session_id", "")
    progress.advance(sid, "writing", 0, "Checking claims against their sources…")
    try:
        out = await registry.grounding_judge(state)
    except Exception as exc:  # noqa: BLE001 — the report is already complete; the judge is an add-on
        _log.warning(f"grounding judge failed: {type(exc).__name__}: {exc}", step="judge_failed", session_id=sid)
        out = {"claim_verdicts": {"verdicts": {}, "summary": {}, "judge": {
            "model": settings.judge_model or "judge", "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:120]}"}}}
    # The Writer snapshotted usage before the judge ran; refresh it so tokens/cost include the judge's calls.
    out["usage_summary"] = tracker.summary(sid)
    return out
