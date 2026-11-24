"""Langfuse prompt management: resolution, fallbacks, placeholder validation, trace linking and sync."""
import asyncio
from types import SimpleNamespace

import pytest

from research_analyst.agents.planner import PlannerAgent
from research_analyst.config import settings
from research_analyst.observability import prompt_store as ps
from research_analyst.observability import prompt_sync
from tests.helpers import FakeLLM, initial_state

LOCAL = "You are a research planning assistant. Local text {min_q}-{max_q}."


class FakeLF:
    """Stands in for the Langfuse client: prompts dict name->(text, version); records calls."""

    def __init__(self, prompts=None, error=None):
        self.prompts, self.error, self.get_calls, self.created = prompts or {}, error, [], []

    def get_prompt(self, name, *, label=None, fallback=None, **kw):
        self.get_calls.append((name, label))
        if self.error:
            raise self.error
        if name in self.prompts:
            text, version = self.prompts[name]
            return SimpleNamespace(prompt=text, version=version, is_fallback=False)
        return SimpleNamespace(prompt=fallback, version=0, is_fallback=True)

    def create_prompt(self, *, name, prompt, labels, **kw):
        self.created.append((name, prompt, labels))
        return SimpleNamespace(version=len(self.created))


@pytest.fixture
def lf(monkeypatch):
    client = FakeLF()
    ps.register("t-plain", "local plain")
    ps.register("t-fmt", "Hello {who}, you have {n}", formatted=True)
    monkeypatch.setattr(ps.tracing, "get_client", lambda: client)
    monkeypatch.setattr(settings, "langfuse_prompts", True)
    return client


def test_local_prompt_when_tracing_disabled(monkeypatch):
    monkeypatch.setattr(ps.tracing, "get_client", lambda: None)
    ps.register("t-off", "x")
    assert ps.resolve("t-off").source == "local" and ps.resolve("t-off").text == "x"


def test_langfuse_text_and_version_are_used_when_found(lf):
    lf.prompts["t-plain"] = ("remote plain", 7)
    r = ps.resolve("t-plain")
    assert (r.text, r.version, r.source, r.label) == ("remote plain", 7, "langfuse", "production")
    assert lf.get_calls == [("t-plain", "production")]
    assert ps.used()["t-plain"] == {"source": "langfuse", "version": 7, "label": "production"}


def test_label_setting_selects_the_version_to_fetch(lf, monkeypatch):
    monkeypatch.setattr(settings, "langfuse_prompt_label", "staging")
    ps.resolve("t-plain")
    assert lf.get_calls[-1] == ("t-plain", "staging")


def test_missing_in_langfuse_falls_back_and_pauses_remote_lookups(lf):
    r = ps.resolve("t-plain")
    assert r.source == "local" and r.text == "local plain"
    ps.resolve("t-fmt")
    assert len(lf.get_calls) == 1                       # second call skipped: remote is considered down for a minute


def test_network_error_falls_back_without_raising(lf):
    lf.error = TimeoutError("slow")
    assert ps.resolve("t-plain").source == "local"


def test_formatted_prompt_with_changed_placeholders_is_rejected(lf):
    lf.prompts["t-fmt"] = ("Hello {who}, you have {count}", 3)          # renamed n -> count in the UI
    r = ps.resolve("t-fmt")
    assert r.source == "local" and r.format(who="a", n=2) == "Hello a, you have 2"


def test_formatted_prompt_with_same_placeholders_is_accepted_and_linked(lf):
    lf.prompts["t-fmt"] = ("Dear {who}: {n} items", 4)
    r = ps.resolve("t-fmt")
    out = r.format(who="Sam", n=3)
    assert out == "Dear Sam: 3 items" and r.version == 4
    assert ps.lookup(out) is r                                          # the rendered system message maps back to the prompt


def test_empty_remote_prompt_is_ignored(lf):
    lf.prompts["t-plain"] = ("   ", 2)
    assert ps.resolve("t-plain").source == "local"


def test_unregistered_name_is_a_clear_error():
    with pytest.raises(KeyError):
        ps.resolve("nope")


def test_langfuse_prompts_flag_off_never_calls_langfuse(lf, monkeypatch):
    monkeypatch.setattr(settings, "langfuse_prompts", False)
    assert ps.resolve("t-plain").source == "local" and lf.get_calls == []


def test_every_system_prompt_of_the_pipeline_is_registered():
    names = set(prompt_sync.load_all())
    assert {"planner-system", "reader-system", "synthesizer-system", "synthesizer-revision-system", "critic-system",
            "writer-system", "query-rewrite-system", "react-search-system", "grounding-judge-system"} <= names


def test_agent_uses_the_remote_prompt_and_links_the_generation(lf, monkeypatch):
    original = ps.registered()["planner-system"]
    remote = original + "\nEXTRA RULE FROM LANGFUSE."
    lf.prompts["planner-system"] = (remote, 5)
    fake = FakeLLM()
    monkeypatch.setattr("research_analyst.llm.generate", fake)
    seen = {}

    from research_analyst.observability import tracing as T
    real = T.observation

    def spy(name, **kw):
        if name == "llm:planner":
            seen.update(kw)
        return real(name, **kw)
    monkeypatch.setattr("research_analyst.agents.base.tracing.observation", spy)
    asyncio.run(PlannerAgent()(initial_state()))
    system = next(p[0]["content"] for role, p in fake.prompts if role == "planner")
    assert "EXTRA RULE FROM LANGFUSE." in system
    assert seen["prompt"].version == 5                                   # generation is linked to Langfuse prompt v5


def test_agent_without_langfuse_sends_the_unchanged_repo_prompt(monkeypatch):
    monkeypatch.setattr(ps.tracing, "get_client", lambda: None)
    fake = FakeLLM()
    monkeypatch.setattr("research_analyst.llm.generate", fake)
    asyncio.run(PlannerAgent()(initial_state()))
    system = next(p[0]["content"] for role, p in fake.prompts if role == "planner")
    assert system.startswith("You are a research planning assistant.")


# ── sync ─────────────────────────────────────────────────────────────────────

def test_sync_creates_missing_prompts_and_skips_identical_ones():
    texts = prompt_sync.load_all()
    client = FakeLF(prompts={"critic-system": (texts["critic-system"], 3)})
    res = {r.name: r for r in prompt_sync.sync(client, "production")}
    assert res["critic-system"].action == "unchanged" and res["critic-system"].version == 3
    assert res["planner-system"].action == "created"
    assert all(labels == ["production"] for _, _, labels in client.created)
    assert "critic-system" not in {n for n, _, _ in client.created}


def test_sync_creates_a_new_version_when_text_differs_and_dry_run_writes_nothing():
    texts = prompt_sync.load_all()
    client = FakeLF(prompts={"writer-system": ("old text", 1)})
    dry = {r.name: r for r in prompt_sync.sync(client, "staging", dry_run=True)}
    assert dry["writer-system"].action == "would-create" and "v1" in dry["writer-system"].detail and client.created == []
    prompt_sync.sync(client, "staging")
    assert ("writer-system", texts["writer-system"], ["staging"]) in client.created


def test_sync_reports_errors_per_prompt_without_stopping():
    class Bad(FakeLF):
        def create_prompt(self, **kw):
            raise RuntimeError("403")
    res = prompt_sync.sync(Bad(), "production")
    assert res and all(r.action == "error" for r in res)
