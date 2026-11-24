from types import SimpleNamespace

import pytest

from research_analyst.config import settings
from research_analyst.llm import preflight


def _fake_client(ids):
    return SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=i) for i in ids])))


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    preflight.reset_cache()
    monkeypatch.setattr(settings, "groq_fast_model", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "groq_strong_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "groq_fallback_model", "qwen/qwen3.8-27b")
    yield
    preflight.reset_cache()


def _patch_models(monkeypatch, ids):
    monkeypatch.setattr("research_analyst.llm.groq_client._get_client", lambda: _fake_client(ids))


def test_all_models_present_is_ok(monkeypatch):
    _patch_models(monkeypatch, ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "whisper-large-v3"])
    c = preflight.check_groq_models()
    assert c.verified and c.ok and not c.fallback_missing
    assert "whisper-large-v3" not in c.suggestions                    # audio models are not offered as chat suggestions
    assert preflight.describe(c) == ("ok", "")


def test_retired_model_is_reported_with_suggestions(monkeypatch):
    _patch_models(monkeypatch, ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "whisper-large-v3"])
    monkeypatch.setattr(settings, "gemini_api_key", "x")
    c = preflight.check_groq_models()
    assert c.missing == ("openai/gpt-oss-20b",)
    level, msg = preflight.describe(c)
    assert level == "warning" and "openai/gpt-oss-20b" in msg and "20 requests/day" in msg and "gpt-oss-120b" in msg


def test_retired_model_without_gemini_key_is_an_error(monkeypatch):
    _patch_models(monkeypatch, ["qwen/qwen3.8-27b"])
    monkeypatch.setattr(settings, "gemini_api_key", "")
    level, msg = preflight.describe(preflight.check_groq_models())
    assert level == "error" and "cannot run" in msg


def test_unreachable_provider_never_blocks(monkeypatch):
    def boom():
        raise OSError("dns")
    monkeypatch.setattr("research_analyst.llm.groq_client._get_client", boom)
    c = preflight.check_groq_models()
    assert not c.verified and c.ok and preflight.describe(c) == ("ok", "")


def test_missing_key_is_unverified_not_an_error(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    assert not preflight.check_groq_models().verified


def test_result_is_cached_for_ten_minutes(monkeypatch):
    calls = {"n": 0}

    def client():
        calls["n"] += 1
        return _fake_client(["openai/gpt-oss-20b", "openai/gpt-oss-120b"])

    monkeypatch.setattr("research_analyst.llm.groq_client._get_client", client)
    preflight.check_groq_models()
    preflight.check_groq_models()
    assert calls["n"] == 1
    preflight.check_groq_models(force=True)
    assert calls["n"] == 2


def test_assert_models_available_raises_before_any_spend(monkeypatch):
    _patch_models(monkeypatch, ["qwen/qwen3.8-27b"])
    with pytest.raises(RuntimeError, match="not available"):
        preflight.assert_models_available()
    preflight.reset_cache()
    _patch_models(monkeypatch, ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
    preflight.assert_models_available()                                # no exception
