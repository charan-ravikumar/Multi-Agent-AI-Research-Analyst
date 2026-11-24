"""
llm/preflight.py — fail fast when a configured model is no longer available.

Why: Groq retired the Llama models this project originally used.  Every call then got a 404, the
client treated it as "provider unavailable" and silently routed everything to the Gemini fallback —
whose free tier allows only ~20 requests/day.  The run "worked" for a while and then died, with an
error far from the cause.  This asks the provider which models exist, once per 10 minutes, and turns
that failure into an explicit, early, actionable message.

Never blocks on its own failure: if the provider can't be reached the check is simply "unverified".
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from research_analyst.config import settings

_NON_CHAT = ("whisper", "guard", "orpheus", "tts", "embed", "moderation")
_TTL_SECONDS = 600.0


@dataclass(frozen=True)
class ModelCheck:
    verified: bool                       # False: provider unreachable / key missing -> do not block
    available: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()        # configured primary models (fast/strong) that are NOT available
    fallback_missing: bool = False       # the second Groq model is configured but gone
    suggestions: tuple[str, ...] = ()    # chat-capable models that ARE available
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing


_cache: tuple[float, ModelCheck] | None = None


def _list_models() -> list[str]:
    from research_analyst.llm import groq_client  # noqa: PLC0415
    client = groq_client._get_client()
    return sorted(m.id for m in client.models.list().data)


def check_groq_models(force: bool = False) -> ModelCheck:
    global _cache
    if not force and _cache and time.monotonic() - _cache[0] < _TTL_SECONDS:
        return _cache[1]
    if not settings.groq_api_key:
        result = ModelCheck(verified=False, error="GROQ_API_KEY is not set")
    else:
        try:
            available = _list_models()
            configured = [settings.groq_fast_model, settings.groq_strong_model]
            missing = tuple(dict.fromkeys(m for m in configured if m not in available))
            fb = settings.groq_fallback_model
            result = ModelCheck(
                verified=True,
                available=tuple(available),
                missing=missing,
                fallback_missing=bool(fb) and fb not in available,
                suggestions=tuple(m for m in available if not any(x in m.lower() for x in _NON_CHAT)),
            )
        except Exception as exc:  # noqa: BLE001 — never let the check itself break the app
            result = ModelCheck(verified=False, error=f"{type(exc).__name__}: {str(exc)[:120]}")
    _cache = (time.monotonic(), result)
    return result


def describe(check: ModelCheck) -> tuple[str, str]:
    """-> (level, message) with level in {"ok", "warning", "error"}; message is user-presentable."""
    if not check.verified or check.ok:
        if check.verified and check.fallback_missing:
            return "warning", f"The fallback Groq model {settings.groq_fallback_model!r} is no longer available; set GROQ_FALLBACK_MODEL."
        return "ok", ""
    pick = ", ".join(check.suggestions[:4]) or "none found"
    base = (f"Configured Groq model(s) not available for this key: {', '.join(check.missing)}. "
            f"Available chat models: {pick}. Set GROQ_FAST_MODEL / GROQ_STRONG_MODEL accordingly.")
    if settings.gemini_api_key:
        return "warning", base + " Until fixed, calls fall back to Gemini, whose free tier allows only ~20 requests/day."
    return "error", base + " No Gemini fallback key is configured, so research cannot run."


def assert_models_available() -> None:
    """For scripts/evaluation: raise before spending any quota if a configured model is gone."""
    level, message = describe(check_groq_models())
    if level in ("warning", "error") and check_groq_models().missing:
        raise RuntimeError(message)


def reset_cache() -> None:
    global _cache
    _cache = None
