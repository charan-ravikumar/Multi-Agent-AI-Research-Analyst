"""
llm/preflight.py — fail fast when a configured Gemini model is not available to this key.

With no fallback provider, a retired or mistyped model name would fail every call.  This asks the API which models
exist (cached 10 minutes) and turns that into an early, actionable message.  It never blocks on its own failure: if
the API can't be reached the check is simply "unverified".
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from research_analyst.config import settings

_TTL_SECONDS = 600.0


@dataclass(frozen=True)
class ModelCheck:
    verified: bool                       # False: API unreachable / key missing -> do not block
    available: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()        # configured models that are NOT available
    suggestions: tuple[str, ...] = ()    # text-generation models that ARE available
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing


_cache: tuple[float, ModelCheck] | None = None


def _list_models() -> list[str]:
    from research_analyst.llm import gemini_client  # noqa: PLC0415
    out = []
    for m in gemini_client._get_client().models.list():
        if "generateContent" in (m.supported_actions or []):
            out.append((m.name or "").removeprefix("models/"))
    return sorted(out)


def configured_models() -> list[str]:
    return list(dict.fromkeys(m for m in (settings.gemini_fast_model, settings.gemini_strong_model, settings.judge_model) if m))


def check_models(force: bool = False) -> ModelCheck:
    global _cache
    if not force and _cache and time.monotonic() - _cache[0] < _TTL_SECONDS:
        return _cache[1]
    if not settings.gemini_api_key:
        result = ModelCheck(verified=False, error="GEMINI_API_KEY is not set")
    else:
        try:
            available = _list_models()
            missing = tuple(m for m in configured_models() if m not in available)
            result = ModelCheck(
                verified=True, available=tuple(available), missing=missing,
                suggestions=tuple(m for m in available if "tts" not in m and "image" not in m and "transcribe" not in m))
        except Exception as exc:  # noqa: BLE001 — never let the check itself break the app
            result = ModelCheck(verified=False, error=f"{type(exc).__name__}: {str(exc)[:120]}")
    _cache = (time.monotonic(), result)
    return result


def describe(check: ModelCheck) -> tuple[str, str]:
    """-> (level, message) with level in {"ok", "error"}; message is user-presentable."""
    if not check.verified or check.ok:
        return "ok", ""
    pick = ", ".join(check.suggestions[:6]) or "none found"
    return "error", (f"Configured Gemini model(s) not available for this key: {', '.join(check.missing)}. "
                     f"Available text models: {pick}. Set GEMINI_FAST_MODEL / GEMINI_STRONG_MODEL accordingly. "
                     "There is no fallback provider, so research cannot run until this is fixed.")


def assert_models_available() -> None:
    """For scripts/evaluation: raise before spending any quota if a configured model is gone."""
    check = check_models()
    level, message = describe(check)
    if level == "error":
        raise RuntimeError(message)


def reset_cache() -> None:
    global _cache
    _cache = None
