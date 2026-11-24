"""Gemini-only LLM layer: retries, limits, quota, breaker, model choice, structured output and function calling."""
import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel

from research_analyst.agents.base import BaseAgent
from research_analyst.config import settings
from research_analyst.llm import ToolCall, ToolSpec, ToolTurn, gemini_client, tool_results_turn, user_turn
from research_analyst.llm import client as c

MSG = [{"role": "user", "content": "x"}]
OK = ("ok", {"total_tokens": 5, "model": "m"})


def _429(retry_after=None):
    e = ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    if retry_after is not None:
        e.response = httpx.Response(429, request=httpx.Request("POST", "http://x"), headers={"retry-after": str(retry_after)})
    return e


def _503():
    return ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})


def _raiser(exc_factory, counter=None):
    def f(*a, **k):
        if counter is not None:
            counter["n"] += 1
        raise exc_factory()
    return f


@pytest.fixture
def sleeps(monkeypatch):
    s = []
    monkeypatch.setattr(c.time, "sleep", lambda x: s.append(x))
    return s


def _gen(**kw):
    return c.generate(MSG, max_tokens=10, return_usage=True, **kw)


# ── retries (no fallback) ────────────────────────────────────────────────────

def test_429_retries_with_backoff_then_fails_without_any_fallback(monkeypatch, sleeps):
    n = {"n": 0}
    monkeypatch.setattr(c.gemini_client, "generate", _raiser(_429, n))
    with pytest.raises(RuntimeError, match="LLM call failed on gemini-3.5-flash-lite"):
        _gen()
    assert n["n"] == c._MAX_RETRIES and len(sleeps) == c._MAX_RETRIES - 1 and all(s > 0 for s in sleeps)


def test_503_is_retried_and_can_recover(monkeypatch, sleeps):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _503()
        return OK
    monkeypatch.setattr(c.gemini_client, "generate", flaky)
    assert _gen()[0] == "ok" and calls["n"] == 3


def test_backoff_has_jitter_and_grows():
    assert len({c._delay(1, _429()) for _ in range(20)}) > 1
    assert max(c._delay(3, _429()) for _ in range(20)) > max(c._delay(1, _429()) for _ in range(20)) * 0.9


def test_retry_after_is_honoured_and_capped():
    assert c._delay(1, _429(retry_after=7)) >= 7
    assert c._delay(1, _429(retry_after=9999)) <= settings.llm_retry_after_cap_seconds + 1e-9


def test_transport_errors_are_retried(monkeypatch, sleeps):
    n = {"n": 0}
    monkeypatch.setattr(c.gemini_client, "generate", _raiser(lambda: OSError("dns"), n))
    with pytest.raises(RuntimeError):
        _gen()
    assert n["n"] == c._MAX_RETRIES


def test_non_retryable_error_is_not_retried(monkeypatch, sleeps):
    n = {"n": 0}
    monkeypatch.setattr(c.gemini_client, "generate",
                        _raiser(lambda: ClientError(400, {"error": {"message": "bad", "status": "INVALID_ARGUMENT"}}), n))
    with pytest.raises(RuntimeError):
        _gen()
    assert n["n"] == 1 and sleeps == []


# ── model choice ─────────────────────────────────────────────────────────────

def test_tiers_map_to_the_two_configured_models_and_override_wins():
    assert c.model_for(c.ModelTier.FAST) == settings.gemini_fast_model == "gemini-3.5-flash-lite"
    assert c.model_for(c.ModelTier.STRONG) == settings.gemini_strong_model == "gemini-3.8-flash"
    assert c.model_for(c.ModelTier.STRONG, "custom-model") == "custom-model"


def test_generate_uses_fast_model_by_default_and_forwards_schema(monkeypatch):
    seen = {}

    class S(BaseModel):
        a: int

    def gen(messages, **k):
        seen.update(k)
        return OK
    monkeypatch.setattr(c.gemini_client, "generate", gen)
    _gen(schema=S, json_mode=True)
    assert seen["model"] == "gemini-3.5-flash-lite" and seen["schema"] is S and seen["json_mode"] is True
    _gen(tier=c.ModelTier.STRONG)
    assert seen["model"] == "gemini-3.8-flash"


def test_only_listed_agents_use_the_strong_model(monkeypatch):
    class A(BaseAgent):
        name = "writer"
        async def run(self, state): return {}

    class B(BaseAgent):
        name = "reader"
        async def run(self, state): return {}
    assert A().tier is c.ModelTier.STRONG and B().tier is c.ModelTier.FAST
    monkeypatch.setattr(settings, "gemini_strong_agents", "writer, critic")
    assert settings.strong_agents == {"writer", "critic"}
    monkeypatch.setattr(settings, "gemini_strong_agents", "")
    assert A().tier is c.ModelTier.FAST


# ── breaker ──────────────────────────────────────────────────────────────────

def test_circuit_breaker_fails_fast_after_repeated_failures(monkeypatch, sleeps):
    n = {"n": 0}
    monkeypatch.setattr(c.gemini_client, "generate", _raiser(_503, n))
    for _ in range(settings.llm_breaker_threshold):
        with pytest.raises(RuntimeError):
            _gen()
    tried = n["n"]
    with pytest.raises(RuntimeError, match="circuit open"):
        _gen()
    assert n["n"] == tried                                      # the provider was not touched while open


def test_breaker_recovers_after_cooldown(monkeypatch, sleeps):
    monkeypatch.setattr(c.gemini_client, "generate", _raiser(_503))
    for _ in range(settings.llm_breaker_threshold):
        with pytest.raises(RuntimeError):
            _gen()
    c._breaker._open_until["gemini-3.5-flash-lite"] = 0.0       # cooldown elapsed
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: OK)
    assert _gen()[0] == "ok" and not c._breaker.is_open("gemini-3.5-flash-lite")


# ── rate limits: RPM / TPM / RPD ─────────────────────────────────────────────

@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(c.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(c.time, "sleep", lambda s: (slept.append(s), now.__setitem__("t", now["t"] + s)))
    return slept


def test_rpm_limiter_waits_for_the_window_to_roll(clock):
    lim = c._RateLimiter()
    for _ in range(5):
        lim.wait_for("m", 10, rpm=5, tpm=0)
    assert clock == []
    lim.wait_for("m", 10, rpm=5, tpm=0)                          # 6th request inside 60 s
    assert clock and 59 < sum(clock) < 62


def test_tpm_limiter_waits_and_settles_to_actual_usage(clock):
    lim = c._RateLimiter()
    lim.wait_for("m", 600, rpm=0, tpm=1000)
    lim.wait_for("m", 600, rpm=0, tpm=1000)                      # 1200 > 1000 -> wait for the first to age out
    assert clock and 59 < sum(clock) < 62
    clock.clear()
    lim2 = c._RateLimiter()
    slot = lim2.wait_for("m", 900, rpm=0, tpm=1000)
    lim2.settle(slot, 100)
    lim2.wait_for("m", 800, rpm=0, tpm=1000)
    assert clock == []


def test_limiter_disabled_when_no_limits():
    c._RateLimiter().wait_for("m", 10**9, rpm=0, tpm=0)


def test_limits_come_from_the_json_setting_and_bad_json_disables(monkeypatch):
    monkeypatch.setattr(settings, "gemini_rate_limits", '{"gemini-3.8-flash": {"rpm": 5, "tpm": 250000, "rpd": 20}}')
    assert c._limits_for("gemini-3.8-flash") == {"rpm": 5, "tpm": 250000, "rpd": 20}
    assert c._limits_for("other") == {}
    monkeypatch.setattr(settings, "gemini_rate_limits", "not json")
    assert c._limits_for("gemini-3.8-flash") == {}


def test_documented_default_limits_match_the_free_tier():
    from research_analyst.config import Settings
    table = json.loads(Settings(_env_file=None, gemini_rate_limits=Settings.model_fields["gemini_rate_limits"].default).gemini_rate_limits)
    assert table["gemini-3.8-flash"] == {"rpm": 5, "tpm": 250000, "rpd": 20}
    assert table["gemini-3.5-flash-lite"] == {"rpm": 15, "tpm": 250000, "rpd": 500}


def test_daily_quota_is_enforced_without_calling_the_api(monkeypatch):
    monkeypatch.setattr(settings, "gemini_rate_limits", '{"gemini-3.5-flash-lite": {"rpd": 2}}')
    n = {"n": 0}

    def gen(*a, **k):
        n["n"] += 1
        return OK
    monkeypatch.setattr(c.gemini_client, "generate", gen)
    _gen()
    _gen()
    with pytest.raises(c.DailyQuotaExceeded, match=r"2/2.*midnight Pacific"):
        _gen()
    assert n["n"] == 2 and c.daily_requests_used("gemini-3.5-flash-lite") == 2


def test_daily_quota_survives_restart_and_resets_on_a_new_day(monkeypatch):
    monkeypatch.setattr(settings, "gemini_rate_limits", '{"m": {"rpd": 1}}')
    c._Quota().reserve("m", 1)
    with pytest.raises(c.DailyQuotaExceeded):
        c._Quota().reserve("m", 1)                               # a fresh object reads the persisted counter
    monkeypatch.setattr(c._Quota, "_today", staticmethod(lambda: "2999-01-01"))
    c._Quota().reserve("m", 1)                                   # new Pacific day: budget is back


def test_each_retry_attempt_counts_against_the_daily_budget(monkeypatch, sleeps):
    monkeypatch.setattr(settings, "gemini_rate_limits", '{"gemini-3.5-flash-lite": {"rpd": 100}}')
    monkeypatch.setattr(c.gemini_client, "generate", _raiser(_503))
    with pytest.raises(RuntimeError):
        _gen()
    assert c.daily_requests_used("gemini-3.5-flash-lite") == c._MAX_RETRIES


def test_daily_quota_error_is_not_retried(monkeypatch, sleeps):
    monkeypatch.setattr(settings, "gemini_rate_limits", '{"gemini-3.5-flash-lite": {"rpd": 1}}')
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: OK)
    _gen()
    with pytest.raises(c.DailyQuotaExceeded):
        _gen()
    assert sleeps == []


def test_estimate_caps_completion_reservation():
    assert c._estimate_tokens(400, 3500) == 100 + c._EXPECTED_COMPLETION


def test_settings_import_without_key(monkeypatch):
    from research_analyst.config import Settings
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert Settings(_env_file=None).gemini_api_key == ""


# ── function calling through the facade ──────────────────────────────────────

TOOLS = [ToolSpec("search_web", "search", {"type": "object", "properties": {"thought": {"type": "string"}, "query": {"type": "string"}},
                                           "required": ["thought", "query"]})]


def test_tool_turn_goes_through_limits_and_returns_the_turn(monkeypatch):
    monkeypatch.setattr(settings, "gemini_rate_limits", '{"gemini-3.5-flash-lite": {"rpd": 1}}')
    turn = ToolTurn(calls=[ToolCall("search_web", {"thought": "t", "query": "q"})], usage={"total_tokens": 9})
    seen = {}

    def tt(system, history, tools, **k):
        seen.update(k, system=system, tools=tools)
        return turn
    monkeypatch.setattr(c.gemini_client, "tool_turn", tt)
    assert c.tool_turn("sys", [user_turn("hi")], TOOLS) is turn
    assert seen["model"] == "gemini-3.5-flash-lite" and seen["system"] == "sys"
    with pytest.raises(c.DailyQuotaExceeded):
        c.tool_turn("sys", [user_turn("hi")], TOOLS)


# ── the Gemini wire layer ────────────────────────────────────────────────────

class _FakeModels:
    def __init__(self, response):
        self.response, self.cfg, self.contents = response, None, None

    def generate_content(self, model, contents, config):
        self.cfg, self.contents, self.model = config, contents, model
        return self.response


def _resp(parts=None, text="{}"):
    content = types.Content(role="model", parts=parts or [types.Part.from_text(text=text)])
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(content=content, finish_reason=None)],
                           usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4, total_token_count=7))


def _patch_client(monkeypatch, response):
    fm = _FakeModels(response)
    monkeypatch.setattr(gemini_client, "_get_client", lambda: SimpleNamespace(models=fm))
    return fm


def test_structured_output_sends_the_schema_and_json_mime_type(monkeypatch):
    class Out(BaseModel):
        x: int
    fm = _patch_client(monkeypatch, _resp(text='{"x": 1}'))
    text, usage = gemini_client.generate(MSG, model="m", schema=Out)
    assert fm.cfg.response_schema is Out and fm.cfg.response_mime_type == "application/json"
    assert text == '{"x": 1}' and usage["provider"] == "gemini" and usage["total_tokens"] == 7


def test_plain_text_call_sets_no_json_mime_type(monkeypatch):
    fm = _patch_client(monkeypatch, _resp(text="hello"))
    gemini_client.generate(MSG, model="m")
    assert fm.cfg.response_mime_type is None and fm.cfg.response_schema is None


def test_thinking_level_is_applied_only_when_configured(monkeypatch):
    fm = _patch_client(monkeypatch, _resp())
    gemini_client.generate(MSG, model="m")
    assert fm.cfg.thinking_config is None                         # thinking_budget=0 is rejected by Gemini 3, so unset
    monkeypatch.setattr(settings, "gemini_thinking_level", "low")
    gemini_client.generate(MSG, model="m")
    assert str(fm.cfg.thinking_config.thinking_level).upper().endswith("LOW")


def test_tool_turn_forces_a_function_call_and_disables_auto_execution(monkeypatch):
    call = types.Part(function_call=types.FunctionCall(name="search_web", args={"thought": "t", "query": "q"}, id="c1"))
    fm = _patch_client(monkeypatch, _resp(parts=[call], text=""))
    turn = gemini_client.tool_turn("sys", [user_turn("go")], TOOLS, model="m")
    fcc = fm.cfg.tool_config.function_calling_config
    assert str(fcc.mode).upper().endswith("ANY") and fcc.allowed_function_names == ["search_web"]
    assert fm.cfg.automatic_function_calling.disable is True
    decl = fm.cfg.tools[0].function_declarations[0]
    assert decl.name == "search_web" and decl.parameters_json_schema["required"] == ["thought", "query"]
    assert [(x.name, x.args, x.id) for x in turn.calls] == [("search_web", {"thought": "t", "query": "q"}, "c1")]
    assert turn.content is not None and turn.usage["total_tokens"] == 7      # content is kept for the next turn


def test_tool_results_turn_answers_every_call_with_its_id():
    t = tool_results_turn([(ToolCall("a", {}, "id1"), {"observation": "x"}), (ToolCall("b", {}, "id2"), {"error": "e"})])
    assert t.role == "user" and [p.function_response.id for p in t.parts] == ["id1", "id2"]
    assert t.parts[0].function_response.response == {"observation": "x"}


def test_empty_candidates_yield_a_turn_without_calls(monkeypatch):
    _patch_client(monkeypatch, SimpleNamespace(text=None, candidates=[], usage_metadata=None))
    turn = gemini_client.tool_turn("s", [user_turn("x")], TOOLS, model="m")
    assert turn.calls == [] and turn.content is None
