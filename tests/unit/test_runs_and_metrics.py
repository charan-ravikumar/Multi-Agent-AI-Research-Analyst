"""Run records, their aggregation, process counters, streaming plumbing and the Monitor page."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from research_analyst.config import settings
from research_analyst.llm import client as c
from research_analyst.llm import gemini_client
from research_analyst.observability import metrics, runs
from research_analyst.observability.progress import progress
from research_analyst.observability.usage import tracker

STATE = {
    "query": "How close are solid-state batteries?", "final_report": "word " * 120, "sub_questions": ["a", "b", "c", "d"],
    "citation_stats": {"report_cited_ratio": 0.9, "report_claim_sentences": 40, "sources_cited": 12, "sources_consulted": 20,
                       "invalid_markers_removed": 2},
    "claim_verdicts": {"summary": {"judged": 20, "supported": 15, "partial": 3, "unsupported": 2}, "judge": {"status": "ok"}},
    "gap_items": [{"kind": "coverage"}, {"kind": "open_question"}, {"kind": "open_question"}],
    "routing_history": ["branch | ok", "branch_failed | [x] boom"], "contradictions": ["c1"],
    "react_traces": [{"stopped": "finish", "steps": [{}, {}]}, {"stopped": "step_cap", "steps": [{}]}],
}
USAGE = {"total_tokens": 22000, "prompt_tokens": 18000, "completion_tokens": 4000, "calls": 24, "truncated_calls": 1,
         "estimated_cost_usd": None,
         "by_agent": {"reader": {"calls": 8, "total_tokens": 9000, "latency_s": 20.5, "prompt_tokens": 1}, "writer": {"calls": 1, "total_tokens": 4000, "latency_s": 30.0}},
         "by_model": {"gemini-3.5-flash-lite": {"calls": 23, "total_tokens": 18000, "latency_s": 60.0}}}


def rec(status="complete", elapsed=130.0, tokens=22000, **kw):
    r = runs.build_record("sid", status, STATE if status in ("complete", "partial") else None, kw.pop("error", ""), elapsed, "q", usage={**USAGE, "total_tokens": tokens})
    r.update(kw)
    return r


# ── building a record ────────────────────────────────────────────────────────

def test_record_flattens_a_finished_run():
    r = rec()
    assert r["status"] == "complete" and r["elapsed_s"] == 130.0 and r["tokens"] == 22000 and r["calls"] == 24
    assert r["cited_ratio"] == 0.9 and r["sources_cited"] == 12 and r["invalid_markers_removed"] == 2
    assert r["judge"] == {"status": "ok", "judged": 20, "supported": 15, "partial": 3, "unsupported": 2}
    assert r["branch_failures"] == 1 and r["coverage_warnings"] == 1 and r["open_questions"] == 2 and r["contradictions"] == 1
    assert r["react"] == {"steps": 3, "stopped": {"finish": 1, "step_cap": 1}}
    assert r["by_agent"]["writer"] == {"calls": 1, "total_tokens": 4000, "latency_s": 30.0}
    assert r["models"]["fast"] == "gemini-3.5-flash-lite" and "writer" in r["models"]["strong_agents"]
    assert r["words"] == 120 and r["run_id"] == "sid" and r["ts"].endswith("+00:00")
    json.dumps(r)                                                   # serialisable as-is


def test_record_stores_no_report_text_and_truncates_the_query():
    r = runs.build_record("s", "complete", STATE, "", 1.0, "q" * 999, usage=USAGE)
    assert len(r["query"]) == 200 and "word word" not in json.dumps(r)


@pytest.mark.parametrize("state,error,cancelled,expected", [
    (STATE, "", False, "partial"), ({"routing_history": ["branch | ok"], "final_report": "x"}, "", False, "complete"),
    (None, "boom", False, "failed"), ({"final_report": "x"}, "", True, "cancelled"), (None, "", False, "failed")])
def test_classify(state, error, cancelled, expected):
    assert runs.classify(state, error, cancelled) == expected


@pytest.mark.parametrize("msg,kind", [
    ("LLM call failed on m after retries: ClientError: 429 RESOURCE_EXHAUSTED", "rate limit"),
    ("Daily request limit for gemini-3.8-flash reached (20/20)", "daily quota"),
    ("ServerError: 503 UNAVAILABLE high demand", "model overloaded"), ("504 DEADLINE_EXCEEDED", "model overloaded"),
    ("LLM circuit open for m", "circuit open"), ("Run exceeded the time limit and was cancelled.", "timeout"),
    ("All 4 research branch(es) failed", "all branches failed"), ("token budget exhausted", "token budget"), ("weird", "other")])
def test_error_kinds(msg, kind):
    assert runs._error_kind(msg) == kind


# ── storage ──────────────────────────────────────────────────────────────────

def test_append_and_load_roundtrip_newest_last_and_skip_corrupt_lines(tmp_path):
    f = str(tmp_path / "r.jsonl")
    for i in range(5):
        runs.append({"run_id": str(i)}, f)
    (tmp_path / "r.jsonl").open("a").write("not json\n")
    assert [r["run_id"] for r in runs.load(3, f)] == ["3", "4"] or [r["run_id"] for r in runs.load(4, f)] == ["2", "3", "4"]
    assert [r["run_id"] for r in runs.load(100, f)] == ["0", "1", "2", "3", "4"]


def test_missing_file_is_an_empty_history_and_write_errors_never_raise(tmp_path):
    assert runs.load(10, str(tmp_path / "nope.jsonl")) == []
    runs.append({"x": 1}, str(tmp_path))                                # a directory: cannot be opened for writing


# ── aggregation ──────────────────────────────────────────────────────────────

def test_summary_headline_numbers():
    records = [rec(elapsed=100), rec(elapsed=200), rec("partial", elapsed=300),
               rec("failed", elapsed=5, error="503 UNAVAILABLE"), rec("cancelled", elapsed=9)]
    s = runs.summarise(records)
    assert (s["runs"], s["counted"], s["complete"], s["partial"], s["failed"], s["cancelled"]) == (5, 4, 2, 1, 1, 1)
    assert s["success_rate"] == 0.75 and s["clean_rate"] == 0.5
    assert s["p50_s"] == 200.0 and s["p95_s"] == 290.0              # failed and cancelled runs do not skew latency
    assert s["median_tokens"] == 22000 and s["errors"] == {"model overloaded": 1}
    assert s["judged_claims"] == 60 and s["supported_share"] == 0.75 and s["unsupported_share"] == 0.1
    assert s["agent_tokens"]["reader"] == 27000 and s["agent_latency_s"]["writer"] == 90.0
    assert s["branch_failures"] == 3 and s["truncated_calls"] == 3


def test_summary_of_nothing_is_all_none_not_a_crash():
    s = runs.summarise([])
    assert s["runs"] == 0 and s["success_rate"] is None and s["p50_s"] is None and s["supported_share"] is None


# ── recording from the UI thread ─────────────────────────────────────────────

def test_research_thread_writes_one_record_per_run(monkeypatch):
    from research_analyst.ui.runner import research_thread

    class App:
        async def ainvoke(self, *a, **k):
            return {**STATE, "routing_history": ["branch | ok"]}
    holder = {"done": False}
    cfg = {"configurable": {"thread_id": "t", "session_id": "sess-1"}}
    research_thread(App(), cfg, ["q"], holder, meta={"query": "my question", "review_rounds": 2})
    got = [r for r in runs.load(50) if r["run_id"] == "sess-1"]
    assert len(got) == 1 and got[0]["status"] == "complete" and got[0]["query"] == "my question" and got[0]["review_rounds"] == 2


def test_failed_and_cancelled_runs_are_recorded_too():
    from research_analyst.ui.runner import research_thread

    class Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("503 UNAVAILABLE")

    class Stop:
        async def ainvoke(self, *a, **k):
            raise asyncio.CancelledError
    for sid, app in (("f1", Boom()), ("c1", Stop())):
        research_thread(app, {"configurable": {"thread_id": "t", "session_id": sid}}, ["q"], {"done": False})
    by = {r["run_id"]: r for r in runs.load(50)}
    assert by["f1"]["status"] == "failed" and by["f1"]["error_kind"] == "model overloaded"
    assert by["c1"]["status"] == "cancelled"


# ── usage latency and process counters ───────────────────────────────────────

def test_usage_tracker_sums_latency_per_agent():
    tracker.record("s", "reader", {"total_tokens": 5, "model": "m", "latency_s": 1.25})
    tracker.record("s", "reader", {"total_tokens": 5, "model": "m", "latency_s": 2.0})
    assert tracker.summary("s")["by_agent"]["reader"]["latency_s"] == 3.25


def test_counters_and_label_breakdown():
    metrics.reset()
    metrics.inc("llm_error", model="m", status=429)
    metrics.inc("llm_error", model="m", status=429)
    metrics.inc("llm_error", model="m", status=503)
    metrics.inc("llm_retry", 1.0, model="m")
    assert metrics.total("llm_error") == 3 and metrics.by_label("llm_error", "status") == {"429": 2.0, "503": 1.0}
    assert metrics.total("llm_retry") == 1.0 and metrics.total("nothing") == 0


def test_client_counts_retries_errors_and_successes(monkeypatch):
    metrics.reset()
    monkeypatch.setattr(c.time, "sleep", lambda s: None)
    from google.genai.errors import ServerError
    n = {"n": 0}

    def flaky(*a, **k):
        n["n"] += 1
        if n["n"] < 3:
            raise ServerError(503, {"error": {"message": "x", "status": "UNAVAILABLE"}})
        return "ok", {"total_tokens": 1}
    monkeypatch.setattr(c.gemini_client, "generate", flaky)
    c.generate([{"role": "user", "content": "x"}], max_tokens=5)
    assert metrics.total("llm_error") == 2 and metrics.total("llm_retry") == 2 and metrics.total("llm_success") == 1


def test_quota_status_lists_every_model_with_limits(monkeypatch):
    monkeypatch.setattr(settings, "gemini_rate_limits", '{"gemini-3.8-flash": {"rpm": 5, "tpm": 250000, "rpd": 20}}')
    c._quota.reserve("gemini-3.8-flash", 20)
    q = {x["model"]: x for x in c.quota_status()}
    assert q["gemini-3.8-flash"] == {"model": "gemini-3.8-flash", "used": 1, "rpd": 20, "rpm": 5, "tpm": 250000}
    assert q["gemini-3.5-flash-lite"]["rpd"] == 0                      # no limit configured -> shown without a bar


# ── streaming ────────────────────────────────────────────────────────────────

def _chunk(text, usage=None, finish=None):
    cand = SimpleNamespace(finish_reason=SimpleNamespace(name=finish) if finish else None, content=None)
    return SimpleNamespace(text=text, candidates=[cand], usage_metadata=usage)


def test_stream_reports_cumulative_text_and_usage(monkeypatch):
    um = SimpleNamespace(prompt_token_count=3, candidates_token_count=5, total_token_count=8)
    chunks = [_chunk("Hello"), _chunk(", wor"), _chunk("ld", um, "STOP")]
    fake = SimpleNamespace(models=SimpleNamespace(generate_content_stream=lambda **kw: iter(chunks)))
    monkeypatch.setattr(gemini_client, "_get_client", lambda: fake)
    seen = []
    text, usage = gemini_client.generate([{"role": "user", "content": "x"}], model="m", on_text=seen.append)
    assert text == "Hello, world" and seen == ["", "Hello", "Hello, wor", "Hello, world"]      # starts empty: retries overwrite
    assert usage["total_tokens"] == 8 and usage["finish_reason"] == "STOP"


def test_stream_survives_a_broken_display_callback(monkeypatch):
    fake = SimpleNamespace(models=SimpleNamespace(generate_content_stream=lambda **kw: iter([_chunk("a"), _chunk("b")])))
    monkeypatch.setattr(gemini_client, "_get_client", lambda: fake)

    def bad(_t):
        raise ValueError("ui gone")
    assert gemini_client.generate([{"role": "user", "content": "x"}], model="m", on_text=lambda t: bad(t) if t else None)[0] == "ab"


def test_progress_snapshot_carries_the_live_text():
    progress.start("live", 2)
    assert progress.snapshot("live")["stream"] == ""
    progress.stream("live", "Partial answer so far")
    assert progress.snapshot("live")["stream"] == "Partial answer so far"
    progress.stream("nobody", "ignored")


def test_the_writer_streams_into_the_progress_tracker(run_graph):
    from tests.helpers import FakeLLM, make_search
    seen = []
    real = progress.stream
    progress.start("will-be-replaced", 1)

    def spy(sid, text):
        seen.append(text)
        real(sid, text)
    import research_analyst.agents.writer as w
    w.progress.stream = spy                                            # type: ignore[method-assign]
    try:
        run_graph(FakeLLM(), make_search(2))
    finally:
        w.progress.stream = real                                       # type: ignore[method-assign]
    assert any("Executive Summary" in t for t in seen)                 # the writer's text reached the live display


# ── the Monitor page ─────────────────────────────────────────────────────────

def _app():
    import os
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "..", "app", "main.py"), default_timeout=60)


def test_monitor_page_renders_empty_state_and_with_data():
    at = _app().run()
    assert not at.exception
    assert any("No runs have been recorded yet" in i.value for i in at.info)
    for i in range(3):
        runs.append(rec("complete", elapsed=100 + i))
    runs.append(rec("failed", elapsed=4, error="503 UNAVAILABLE"))
    at = _app().run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Runs recorded"] == "4" and labels["Completed"] == "75%" and labels["Claims supported"] == "75%"
    assert not any("No runs have been recorded yet" in i.value for i in at.info)
