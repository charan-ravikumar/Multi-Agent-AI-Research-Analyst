import json

import pytest

from config import settings
from core.usage import BudgetExceeded, estimate_cost, tracker
from tests.helpers import FakeLLM, make_search


def u(model="m", p=100, c=50, fr="stop"):
    return {"model": model, "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c, "finish_reason": fr}


def test_tracker_aggregates_by_agent_and_model():
    tracker.record("s", "reader", u("small", 100, 10))
    tracker.record("s", "reader", u("small", 200, 20))
    tracker.record("s", "writer", u("big", 300, 300, fr="length"))
    s = tracker.summary("s")
    assert s["calls"] == 3 and s["total_tokens"] == 110 + 220 + 600
    assert s["by_agent"]["reader"]["calls"] == 2 and s["by_model"]["big"]["completion_tokens"] == 300
    assert s["truncated_calls"] == 1 and s["estimated_cost_usd"] is None


def test_cost_estimate_from_price_table(monkeypatch):
    monkeypatch.setattr(settings, "llm_price_per_1m_tokens", json.dumps({"small": [1.0, 2.0]}))
    tracker.record("s", "a", u("small", 1_000_000, 500_000))
    tracker.record("s", "a", u("other", 10, 10))
    s = tracker.summary("s")
    assert s["estimated_cost_usd"] == pytest.approx(2.0) and s["unpriced_models"] == ["other"]
    assert estimate_cost(u("small", 1_000_000, 0)) == pytest.approx(1.0) and estimate_cost(u("zzz")) is None


def test_bad_price_table_is_ignored(monkeypatch):
    monkeypatch.setattr(settings, "llm_price_per_1m_tokens", "not json")
    tracker.record("s", "a", u())
    assert tracker.summary("s")["estimated_cost_usd"] is None


def test_budget_check(monkeypatch):
    monkeypatch.setattr(settings, "token_budget_per_run", 100)
    tracker.check("s")                         # nothing spent yet
    tracker.record("s", "a", u(p=60, c=40))
    with pytest.raises(BudgetExceeded):
        tracker.check("s")
    tracker.check("other-session")             # budgets are per session


def test_sessions_are_bounded():
    for i in range(tracker._MAX_SESSIONS + 20):
        tracker.record(f"s{i}", "a", u())
    assert len(tracker._s) == tracker._MAX_SESSIONS


def test_run_reports_usage_summary(run_graph):
    res = run_graph(FakeLLM(4), make_search(2))
    s = res["usage_summary"]
    assert s["calls"] == 1 + 4 + 4 + 1 + 1 and s["by_agent"]["reader"]["calls"] == 4
    assert tracker.total(res["session_id"]) == 0          # popped after the run (memory node)


def test_budget_exhaustion_degrades_instead_of_failing(run_graph, monkeypatch):
    # fake usage is 2 tokens/call: planner(2)+4 readers+4 synth = 18 tokens before the debate starts
    monkeypatch.setattr(settings, "token_budget_per_run", 18)
    res = run_graph(FakeLLM(4), make_search(2))
    assert res["debate_budget_exhausted"] is True and res["final_report"]
    assert any("BUDGET_EXHAUSTED" in r for r in res["routing_history"])


def test_budget_exhausted_before_branches_fails_run_clearly(run_graph, monkeypatch):
    monkeypatch.setattr(settings, "token_budget_per_run", 2)       # planner spends it all
    with pytest.raises(RuntimeError, match="research branch"):
        run_graph(FakeLLM(4), make_search(2))


def test_writer_note_when_budget_exhausted(run_graph, monkeypatch):
    monkeypatch.setattr(settings, "token_budget_per_run", 18)
    fake = FakeLLM(4)
    run_graph(fake, make_search(2))
    writer_prompt = [p for r, p in fake.prompts if r == "writer"][0][1]["content"]
    assert "token budget was exhausted" in writer_prompt
