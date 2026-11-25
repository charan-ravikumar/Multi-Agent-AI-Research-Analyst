"""Pieces that make re-running a step safe."""
import research_analyst.memory.redis_store as rs
from research_analyst.config import settings
from research_analyst.observability.usage import BudgetExceeded, tracker


def test_saving_the_same_run_twice_keeps_one_record():
    for report in ("first", "second"):
        rs.save_report_record(session_id="run-1", query="solid state batteries", final_report=report, sub_questions=[])
    [record] = rs.get_recent_reports()
    assert record["executive_summary"] == "second"


def test_usage_exported_by_one_attempt_still_counts_against_the_budget_in_the_next(monkeypatch):
    tracker.record("r", "reader", {"model": "m", "prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120})
    exported = tracker.export("r")
    tracker.reset()
    assert tracker.export("r") is None
    tracker.restore("r", exported)
    tracker.restore("r", exported)                       # restoring is a replacement, not an addition
    assert tracker.total("r") == 120 and tracker.summary("r")["by_agent"]["reader"]["calls"] == 1
    monkeypatch.setattr(settings, "token_budget_per_run", 100)
    try:
        tracker.check("r")
    except BudgetExceeded:
        return
    raise AssertionError("the restored usage should have exhausted the budget")
