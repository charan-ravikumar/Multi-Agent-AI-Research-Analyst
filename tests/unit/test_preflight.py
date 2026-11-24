from types import SimpleNamespace

import pytest

from research_analyst.config import settings
from research_analyst.llm import preflight


def _model(name, actions=("generateContent",)):
    return SimpleNamespace(name=f"models/{name}", supported_actions=list(actions))


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    preflight.reset_cache()
    monkeypatch.setattr(settings, "gemini_api_key", "x")
    monkeypatch.setattr(settings, "gemini_fast_model", "gemini-3.5-flash-lite")
    monkeypatch.setattr(settings, "gemini_strong_model", "gemini-3.8-flash")
    monkeypatch.setattr(settings, "judge_model", "")
    yield
    preflight.reset_cache()


def _patch_models(monkeypatch, models):
    fake = SimpleNamespace(models=SimpleNamespace(list=lambda: models))
    monkeypatch.setattr("research_analyst.llm.gemini_client._get_client", lambda: fake)


def test_all_models_present_is_ok(monkeypatch):
    _patch_models(monkeypatch, [_model("gemini-3.5-flash-lite"), _model("gemini-3.8-flash"),
                                _model("gemini-3.8-flash-tts"), _model("embed", ("embedContent",))])
    c = preflight.check_models()
    assert c.verified and c.ok
    assert "embed" not in c.available                                  # non-generation models are ignored
    assert "gemini-3.8-flash-tts" not in c.suggestions                 # nor offered as suggestions
    assert preflight.describe(c) == ("ok", "")


def test_missing_model_is_an_error_naming_it_and_the_alternatives(monkeypatch):
    _patch_models(monkeypatch, [_model("gemini-3.5-flash-lite"), _model("gemini-3.7-flash")])
    c = preflight.check_models()
    assert c.missing == ("gemini-3.8-flash",)
    level, msg = preflight.describe(c)
    assert level == "error" and "gemini-3.8-flash" in msg and "gemini-3.7-flash" in msg and "no fallback" in msg


def test_a_configured_judge_model_is_checked_too(monkeypatch):
    monkeypatch.setattr(settings, "judge_model", "gemini-9-ghost")
    _patch_models(monkeypatch, [_model("gemini-3.5-flash-lite"), _model("gemini-3.8-flash")])
    assert preflight.check_models().missing == ("gemini-9-ghost",)


def test_unreachable_api_never_blocks(monkeypatch):
    def boom():
        raise OSError("dns")
    monkeypatch.setattr("research_analyst.llm.gemini_client._get_client", boom)
    c = preflight.check_models()
    assert not c.verified and c.ok and preflight.describe(c) == ("ok", "")


def test_missing_key_is_unverified_not_an_error(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert not preflight.check_models().verified


def test_result_is_cached_for_ten_minutes(monkeypatch):
    calls = {"n": 0}

    def client():
        calls["n"] += 1
        return SimpleNamespace(models=SimpleNamespace(list=lambda: [_model("gemini-3.5-flash-lite"), _model("gemini-3.8-flash")]))
    monkeypatch.setattr("research_analyst.llm.gemini_client._get_client", client)
    preflight.check_models()
    preflight.check_models()
    assert calls["n"] == 1
    preflight.check_models(force=True)
    assert calls["n"] == 2


def test_assert_models_available_raises_before_any_spend(monkeypatch):
    _patch_models(monkeypatch, [_model("gemini-3.5-flash-lite")])
    with pytest.raises(RuntimeError, match="not available"):
        preflight.assert_models_available()
    preflight.reset_cache()
    _patch_models(monkeypatch, [_model("gemini-3.5-flash-lite"), _model("gemini-3.8-flash")])
    preflight.assert_models_available()
