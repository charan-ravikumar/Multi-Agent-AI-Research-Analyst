"""Langfuse wiring, tested against a fake client (no network, no real keys)."""
from contextlib import contextmanager

import pytest

from research_analyst.config import settings
from research_analyst.observability import tracing
from tests.helpers import FakeLLM, make_search


class FakeObs:
    def __init__(self, rec):
        self.rec = rec
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)


class FakeLangfuse:
    def __init__(self, fail_start=False):
        self.observations = []     # (name, as_type, kwargs, FakeObs)
        self.flushed = 0
        self.fail_start = fail_start

    @contextmanager
    def start_as_current_observation(self, *, name, as_type="span", **kw):
        if self.fail_start:
            raise RuntimeError("langfuse down")
        o = FakeObs(self)
        self.observations.append((name, as_type, kw, o))
        yield o

    def flush(self):
        self.flushed += 1


@pytest.fixture
def fake_lf(monkeypatch):
    lf = FakeLangfuse()
    sessions = []

    @contextmanager
    def propagate(**kw):
        sessions.append(kw.get("session_id"))
        yield

    tracing._client, tracing._init_done, tracing._propagate = lf, True, propagate
    lf.sessions = sessions
    return lf


def test_disabled_by_default_makes_no_client():
    assert tracing.enabled() is False and tracing._get_client() is None
    with tracing.observation("x") as h:
        h.update(output="ignored")          # no-op, no error


def test_enabled_requires_both_keys(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    assert tracing.enabled() is False
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    assert tracing.enabled() is True


def test_full_run_emits_nested_agent_and_generation_observations(fake_lf, run_graph):
    res = run_graph(FakeLLM(4), make_search(2))
    names = [(n, t) for n, t, _, _ in fake_lf.observations]
    assert ("agent:planner", "agent") in names and ("agent:writer", "agent") in names
    gens = [(n, kw, o) for n, t, kw, o in fake_lf.observations if t == "generation"]
    assert len(gens) == res["usage_summary"]["calls"]
    _, kw, obs = gens[0]
    upd = obs.updates[0]
    assert upd["usage_details"]["total"] == 2 and upd["model"] == "fake" and upd["metadata"]["provider"] == "fake"
    assert set(fake_lf.sessions) == {res["session_id"]}


def test_capture_io_false_sends_no_prompts(fake_lf, run_graph, monkeypatch):
    monkeypatch.setattr(settings, "langfuse_capture_io", False)
    run_graph(FakeLLM(4), make_search(2))
    gen = next(kw for n, t, kw, o in fake_lf.observations if t == "generation")
    assert gen["input"] is None
    assert all(u.get("output") is None for *_, o in fake_lf.observations for u in o.updates)


def test_io_is_truncated(fake_lf, monkeypatch):
    monkeypatch.setattr(settings, "langfuse_max_io_chars", 10)
    with tracing.observation("x", input="a" * 100):
        pass
    sent = fake_lf.observations[0][2]["input"]
    assert sent.startswith("a" * 10) and "+90 chars" in sent


def test_exception_is_recorded_and_reraised(fake_lf):
    with pytest.raises(ValueError), tracing.observation("boom"):
        raise ValueError("bad")
    upd = fake_lf.observations[0][3].updates[-1]
    assert upd["level"] == "ERROR" and "ValueError" in upd["status_message"]


def test_tracing_failure_never_breaks_the_run(run_graph):
    tracing._client, tracing._init_done, tracing._propagate = FakeLangfuse(fail_start=True), True, None
    res = run_graph(FakeLLM(4), make_search(2))
    assert res["final_report"]


def test_run_span_flushes(fake_lf):
    with tracing.run_span("research", "sess-1", "q"):
        pass
    assert fake_lf.flushed == 1 and fake_lf.observations[0][0] == "run:research"
    assert fake_lf.sessions == ["sess-1"]


def test_bad_credentials_disable_tracing_once(monkeypatch):
    import langfuse
    calls = {"n": 0}

    class Rejecting:
        def __init__(self, **kw):
            pass

        def auth_check(self):
            calls["n"] += 1
            return False

    monkeypatch.setattr(langfuse, "Langfuse", Rejecting)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    tracing.reset_for_tests()
    assert tracing._get_client() is None and tracing._get_client() is None
    assert calls["n"] == 1
    with tracing.observation("x"):        # still safe to use
        pass


def test_base_url_takes_precedence_over_host(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_host", "https://cloud.langfuse.com")
    assert settings.langfuse_url == "https://cloud.langfuse.com"
    monkeypatch.setattr(settings, "langfuse_base_url", "https://us.cloud.langfuse.com/")
    assert settings.langfuse_url == "https://us.cloud.langfuse.com"


def test_run_scores_extracts_quality_and_cost_metrics():
    state = {"final_report": "one two three", "routing_history": ["branch_failed | x", "ok"], "debate_forced": True,
             "citation_stats": {"cited_ratio": 0.8, "uncited": 2, "weakly_supported": 1, "sources_cited": 5},
             "usage_summary": {"total_tokens": 1000, "calls": 7, "estimated_cost_usd": 0.01},
             "gap_items": [{"kind": "coverage"}, {"kind": "open_question"}, {"kind": "open_question"}]}
    s = tracing.run_scores(state)
    assert s["cited_ratio"] == 0.8 and s["total_tokens"] == 1000 and s["llm_calls"] == 7 and s["estimated_cost_usd"] == 0.01
    assert s["branch_failures"] == 1 and s["coverage_warnings"] == 1 and s["open_questions"] == 2
    assert s["review_loop_forced"] == 1.0 and s["report_words"] == 3


def test_record_scores_pushes_numeric_scores_to_the_current_trace(fake_lf):
    scores = []
    fake_lf.create_score = lambda **kw: scores.append(kw)
    fake_lf.get_current_trace_id = lambda: "trace-123"
    tracing.record_scores({"final_report": "a b", "citation_stats": {"cited_ratio": 0.5}})
    names = {s["name"] for s in scores}
    assert {"cited_ratio", "report_words"} <= names
    assert all(s["trace_id"] == "trace-123" and s["data_type"] == "NUMERIC" for s in scores)


def test_record_scores_is_silent_when_disabled_or_broken(fake_lf):
    tracing.reset_for_tests()
    tracing.record_scores({"final_report": "x"})                 # disabled: no client, no error
    tracing._client, tracing._init_done = fake_lf, True
    fake_lf.create_score = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    fake_lf.get_current_trace_id = lambda: "t"
    tracing.record_scores({"final_report": "x"})                 # a scoring failure never propagates


def test_run_span_tags_and_prompt_versions(fake_lf, run_graph):
    from tests.helpers import FakeLLM, make_search
    with tracing.run_span("research", "sess-t", "q", tags=["review:off"]):
        pass
    obs = fake_lf.observations[0]
    assert obs[0] == "run:research"
    run_graph(FakeLLM(4), make_search(2))
    gens = [kw for n, t, kw, o in fake_lf.observations if t == "generation"]
    versions = {kw.get("version") for kw in gens}
    assert None not in versions and all(len(v) == 8 for v in versions)
    assert len(versions) >= 4                                    # planner/reader/synth/writer prompts differ
