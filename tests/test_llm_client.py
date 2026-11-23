"""Retry / fallback / breaker / limiter behaviour of llm/client.py with both providers mocked."""
import groq
import httpx
import pytest
from google.genai.errors import ClientError

from config import settings
from llm import client as c

REQ = httpx.Request("POST", "http://x")
MSG = [{"role": "user", "content": "x"}]


def _rate_limit(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return groq.RateLimitError("429", response=httpx.Response(429, request=REQ, headers=headers), body=None)


def _conn_err():
    return groq.APIConnectionError(request=REQ)


def _raiser(exc_factory, counter=None):
    def f(*a, **k):
        if counter is not None:
            counter["n"] += 1
        raise exc_factory()
    return f


def _call(**kw):
    return c._call_with_fallback(MSG, tier=c.ModelTier.FAST, temperature=0.2, max_tokens=10, **kw)


@pytest.fixture
def sleeps(monkeypatch):
    s = []
    monkeypatch.setattr(c.time, "sleep", lambda x: s.append(x))
    return s


def test_groq_429_retries_with_backoff_then_falls_back(monkeypatch, sleeps):
    n = {"n": 0}
    monkeypatch.setattr(c.groq_client, "generate", _raiser(_rate_limit, n))
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: ("from-gemini", {}))
    assert _call()[0] == "from-gemini"
    assert n["n"] == c._MAX_RETRIES
    assert len(sleeps) == c._MAX_RETRIES - 1            # no sleep after the final attempt
    assert all(s > 0 for s in sleeps)


def test_backoff_has_jitter_and_grows(monkeypatch):
    d1 = {c._delay(1, _rate_limit()) for _ in range(20)}
    assert len(d1) > 1                                   # jittered, not constant
    assert max(c._delay(3, _rate_limit()) for _ in range(20)) > max(d1) * 0.9


def test_retry_after_is_honoured_and_capped(monkeypatch):
    assert c._delay(1, _rate_limit(retry_after=7)) >= 7
    assert c._delay(1, _rate_limit(retry_after=9999)) <= settings.llm_retry_after_cap_seconds + 1e-9


def test_connection_errors_are_retried(monkeypatch, sleeps):
    n = {"n": 0}
    monkeypatch.setattr(c.groq_client, "generate", _raiser(_conn_err, n))
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: ("ok", {}))
    assert _call()[0] == "ok"
    assert n["n"] == c._MAX_RETRIES


def test_non_retryable_error_skips_retries(monkeypatch, sleeps):
    n = {"n": 0}
    bad_key = lambda: groq.AuthenticationError("401", response=httpx.Response(401, request=REQ), body=None)  # noqa: E731
    monkeypatch.setattr(c.groq_client, "generate", _raiser(bad_key, n))
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: ("ok", {}))
    assert _call()[0] == "ok"
    assert n["n"] == 1 and sleeps == []


def test_json_mode_and_prefer_are_forwarded(monkeypatch):
    seen = {}

    def groq_gen(messages, **k):
        seen["groq"] = k
        return "{}", {}

    def gem_gen(messages, **k):
        seen["gemini"] = k
        return "{}", {}

    monkeypatch.setattr(c.groq_client, "generate", groq_gen)
    monkeypatch.setattr(c.gemini_client, "generate", gem_gen)
    _call(json_mode=True)
    assert seen["groq"]["json_mode"] is True and "gemini" not in seen
    seen.clear()
    _call(prefer="gemini")
    assert "gemini" in seen and "groq" not in seen


def test_total_failure_is_runtime_error_with_both_reasons(monkeypatch, sleeps):
    monkeypatch.setattr(c.groq_client, "generate", _raiser(_conn_err))
    monkeypatch.setattr(c.gemini_client, "generate", _raiser(lambda: OSError("dns")))
    with pytest.raises(RuntimeError, match="All LLM providers failed") as ei:
        _call()
    assert "groq" in str(ei.value) and "gemini" in str(ei.value)


def test_gemini_429_exhaustion_is_runtime_error(monkeypatch, sleeps):
    monkeypatch.setattr(c.groq_client, "generate", _raiser(_conn_err))
    monkeypatch.setattr(c.gemini_client, "generate",
                        _raiser(lambda: ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})))
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        _call()


def test_circuit_breaker_skips_dead_provider(monkeypatch, sleeps):
    n = {"n": 0}
    monkeypatch.setattr(c.groq_client, "generate", _raiser(_conn_err, n))
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: ("ok", {}))
    for _ in range(settings.llm_breaker_threshold):
        _call()
    tried = n["n"]
    assert tried == settings.llm_breaker_threshold * c._MAX_RETRIES
    _call()                                                # breaker open: Groq not touched
    assert n["n"] == tried and c._breaker.is_open("groq")


def test_breaker_recovers_after_cooldown(monkeypatch, sleeps):
    monkeypatch.setattr(c.groq_client, "generate", _raiser(_conn_err))
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: ("ok", {}))
    for _ in range(settings.llm_breaker_threshold):
        _call()
    assert c._breaker.is_open("groq")
    monkeypatch.setattr(settings, "llm_breaker_cooldown_seconds", 0.0)
    c._breaker._open_until["groq"] = 0.0                   # cooldown elapsed
    monkeypatch.setattr(c.groq_client, "generate", lambda *a, **k: ("groq-back", {}))
    assert _call()[0] == "groq-back" and not c._breaker.is_open("groq")


def test_tpm_limiter_waits_when_window_full(monkeypatch):
    now = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(c.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(c.time, "sleep", lambda s: (slept.append(s), now.__setitem__("t", now["t"] + s)))
    lim = c._TpmLimiter()
    lim.wait_for("m", 600, 1000)      # fits
    assert slept == []
    lim.wait_for("m", 600, 1000)      # 1200 > 1000 -> must wait for first entry to age out
    assert slept and 59 < sum(slept) < 62


def test_tpm_limiter_settles_to_actual_usage(monkeypatch):
    now = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(c.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(c.time, "sleep", lambda s: (slept.append(s), now.__setitem__("t", now["t"] + s)))
    lim = c._TpmLimiter()
    slot = lim.wait_for("m", 900, 1000)
    lim.settle(slot, 100)                       # call actually used 100 tokens, not the 900 reserved
    lim.wait_for("m", 800, 1000)                # fits immediately now
    assert slept == []


def test_estimate_caps_completion_reservation():
    big = [{"role": "user", "content": "x" * 400}]
    assert c._estimate_tokens(big, 3500) == 100 + c._EXPECTED_COMPLETION


def test_tpm_limiter_disabled_by_default():
    c._TpmLimiter().wait_for("m", 10**9, 0)                # budget 0 -> never blocks


def test_settings_import_without_keys(monkeypatch):
    from config import Settings
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.groq_api_key == "" and s.gemini_api_key == ""


def test_gemini_thinking_is_disabled_by_default():
    from google.genai import types
    from llm import gemini_client
    captured = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured["cfg"] = config
            raise RuntimeError("stop")

    class FakeClient:
        models = FakeModels()

    import pytest
    from unittest.mock import patch
    with patch.object(gemini_client, "_get_client", lambda: FakeClient()), pytest.raises(RuntimeError):
        gemini_client.generate([{"role": "user", "content": "x"}], json_mode=True)
    assert captured["cfg"].thinking_config.thinking_budget == 0
    assert isinstance(captured["cfg"].thinking_config, types.ThinkingConfig)


def test_groq_fallback_model_used_before_gemini_on_status_errors(monkeypatch, sleeps):
    monkeypatch.setattr(settings, "groq_fallback_model", "qwen/test")
    seen = []

    def groq_gen(messages, *, model, **k):
        seen.append(model)
        if model != "qwen/test":
            raise _rate_limit()
        return "from-qwen", {"model": model}

    monkeypatch.setattr(c.groq_client, "generate", groq_gen)
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gemini must not be touched")))
    assert _call()[0] == "from-qwen"
    assert seen.count("qwen/test") == 1 and len(seen) == c._MAX_RETRIES + 1


def test_groq_fallback_skipped_when_groq_is_unreachable(monkeypatch, sleeps):
    monkeypatch.setattr(settings, "groq_fallback_model", "qwen/test")
    n = {"n": 0}
    monkeypatch.setattr(c.groq_client, "generate", _raiser(_conn_err, n))
    monkeypatch.setattr(c.gemini_client, "generate", lambda *a, **k: ("gem", {}))
    assert _call()[0] == "gem" and n["n"] == c._MAX_RETRIES          # no second Groq model attempt


def test_model_override_selects_groq_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(c.groq_client, "generate", lambda m, *, model, **k: (seen.setdefault("m", model), ("{}", {}))[1])
    c._call_with_fallback(MSG, tier=c.ModelTier.STRONG, temperature=0.2, max_tokens=10, groq_model_override="qwen/x")
    assert seen["m"] == "qwen/x"
