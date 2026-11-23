"""
llm/client.py — INTERNAL provider orchestration layer

Do not import this module directly from agents or tools.  Use the facade:

    import llm
    text = llm.generate("Explain AI")
    text, usage = llm.generate("...", return_usage=True, json_mode=True)

Policy (single place, SDK-level retries are disabled so attempts don't multiply):
  * retryable errors (429, 5xx, timeouts, connection errors) are retried with
    exponential backoff + jitter, honouring Retry-After (capped); never sleeping
    after the final attempt.  Non-retryable errors (400/401/403/404...) fall
    straight through to the other provider.
  * per-provider circuit breaker: after N consecutive exhausted calls the
    provider is skipped for a cooldown.
  * optional per-model sliding-window TPM budget (settings.llm_tpm_budget).
  * provider order: Groq -> Gemini by default; ``prefer="gemini"`` reverses it.
  * every terminal failure is a ``RuntimeError("All LLM providers failed ...")``.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, List, Literal, Union, overload

import groq
from google.genai.errors import APIError as GeminiAPIError

from config import settings
from llm import gemini_client, groq_client

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = settings.llm_max_retries
_BACKOFF_BASE: float = settings.llm_backoff_base


class ModelTier(str, Enum):
    """FAST -> cheap/quick (Reader); STRONG -> reasoning agents (Planner, Synthesizer, Critic, Writer)."""
    FAST = "fast"
    STRONG = "strong"


_GROQ_TIER_MAP = {
    ModelTier.FAST: groq_client.GROQ_FAST_MODEL,
    ModelTier.STRONG: groq_client.GROQ_STRONG_MODEL,
}
_GEMINI_TIER_MAP = {
    ModelTier.FAST: gemini_client.GEMINI_MODEL,
    ModelTier.STRONG: gemini_client.GEMINI_MODEL,
}


# ── public API ───────────────────────────────────────────────────────────────

@overload
def generate(prompt: Union[str, List[dict]], *, tier: ModelTier = ..., system: str | None = ...,
             temperature: float | None = ..., max_tokens: int | None = ..., return_usage: Literal[False] = ...,
             json_mode: bool = ..., prefer: str | None = ..., groq_model: str | None = ...) -> str: ...
@overload
def generate(prompt: Union[str, List[dict]], *, tier: ModelTier = ..., system: str | None = ...,
             temperature: float | None = ..., max_tokens: int | None = ..., return_usage: Literal[True],
             json_mode: bool = ..., prefer: str | None = ..., groq_model: str | None = ...) -> tuple[str, dict]: ...
def generate(
    prompt: Union[str, List[dict]],
    *,
    tier: ModelTier = ModelTier.STRONG,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    return_usage: bool = False,
    json_mode: bool = False,
    prefer: str | None = None,
    groq_model: str | None = None,
) -> str | tuple[str, dict]:
    """
    Generate a response.  ``json_mode`` asks the provider for a valid JSON object;
    ``prefer="gemini"`` tries Gemini first (used for the Critic).
    Returns text, or (text, usage) when ``return_usage`` is True.
    """
    _temperature = temperature if temperature is not None else settings.llm_temperature
    _max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    messages = _build_messages(prompt, system)
    text, usage = _call_with_fallback(
        messages, tier=tier, temperature=_temperature, max_tokens=_max_tokens,
        json_mode=json_mode, prefer=prefer, groq_model_override=groq_model,
    )
    return (text, usage) if return_usage else text


def _build_messages(prompt: Union[str, List[dict]], system: str | None) -> List[dict]:
    if isinstance(prompt, list):
        return prompt
    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


# ── circuit breaker ──────────────────────────────────────────────────────────

class _Breaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def is_open(self, provider: str) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until.get(provider, 0.0)

    def success(self, provider: str) -> None:
        with self._lock:
            self._fails[provider] = 0
            self._open_until.pop(provider, None)

    def failure(self, provider: str) -> None:
        with self._lock:
            n = self._fails.get(provider, 0) + 1
            self._fails[provider] = n
            if n >= settings.llm_breaker_threshold:
                self._open_until[provider] = time.monotonic() + settings.llm_breaker_cooldown_seconds
                logger.warning("circuit OPEN for %s (%d consecutive failures, cooldown %.0fs)",
                               provider, n, settings.llm_breaker_cooldown_seconds)

    def reset(self) -> None:
        with self._lock:
            self._fails.clear()
            self._open_until.clear()


_breaker = _Breaker()


# ── sliding-window TPM budget ────────────────────────────────────────────────

class _TpmLimiter:
    """Blocks (in the calling worker thread) until the model's 60s window has room.

    A call first *reserves* an estimate, then ``settle`` replaces it with the actual
    token count, so over-estimates don't throttle later calls for a full minute."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._win: dict[str, deque] = {}

    def wait_for(self, model: str, est_tokens: int, budget: int) -> list | None:
        if budget <= 0:
            return None
        est_tokens = min(est_tokens, budget)       # a single call larger than the budget must still run
        while True:
            with self._lock:
                q = self._win.setdefault(model, deque())
                now = time.monotonic()
                while q and now - q[0][0] >= 60.0:
                    q.popleft()
                used = sum(e[1] for e in q)
                if used + est_tokens <= budget:
                    entry = [now, est_tokens]
                    q.append(entry)
                    return entry
                wait = 60.0 - (now - q[0][0]) + 0.05
            logger.info("TPM budget: waiting %.1fs for %s", wait, model)
            time.sleep(max(wait, 0.05))

    @staticmethod
    def settle(entry: list | None, actual_tokens: int | None) -> None:
        if entry is not None and actual_tokens:
            entry[1] = int(actual_tokens)

    def reset(self) -> None:
        with self._lock:
            self._win.clear()


_tpm = _TpmLimiter()

_EXPECTED_COMPLETION = 600   # reservation cap: most calls finish far below max_tokens


def _estimate_tokens(messages: List[dict], max_tokens: int) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4 + min(max_tokens, _EXPECTED_COMPLETION)


# ── retry classification ─────────────────────────────────────────────────────

def _status_of(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):   # timeout is a subclass of connection error
        return True
    code = _status_of(exc)
    if code is not None:
        return code == 429 or code == 408 or code >= 500
    # transport-level errors (httpx, socket, ...) from either SDK
    return isinstance(exc, (OSError, TimeoutError)) or type(exc).__module__.startswith("httpx")


def _retry_after(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _delay(attempt: int, exc: BaseException) -> float:
    backoff = (_BACKOFF_BASE ** attempt) * random.uniform(0.5, 1.0)       # jittered exponential
    ra = _retry_after(exc)
    if ra is not None:
        backoff = max(backoff, min(ra, settings.llm_retry_after_cap_seconds))
    return backoff


def _with_retries(provider: str, model: str, call: Callable[[], tuple[str, dict]]) -> tuple[str, dict]:
    """Run ``call`` under the retry policy.  Raises the last exception when exhausted/non-retryable."""
    last: BaseException | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — classified below
            last = exc
            if not _retryable(exc):
                logger.warning("%s non-retryable error: %s: %s", provider, type(exc).__name__, exc)
                break
            if attempt == _MAX_RETRIES:
                logger.warning("%s retries exhausted (%d/%d): %s", provider, attempt, _MAX_RETRIES, exc)
                break
            wait = _delay(attempt, exc)
            logger.warning("%s retryable error (attempt %d/%d) — waiting %.1fs: %s",
                           provider, attempt, _MAX_RETRIES, wait, exc)
            time.sleep(wait)
    assert last is not None
    raise last


# ── orchestration ────────────────────────────────────────────────────────────

def _call_with_fallback(
    messages: List[dict],
    *,
    tier: ModelTier,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
    prefer: str | None = None,
    groq_model_override: str | None = None,
) -> tuple[str, dict]:
    groq_model = groq_model_override or _GROQ_TIER_MAP[tier]
    gemini_model = _GEMINI_TIER_MAP[tier]

    def run_gemini() -> tuple[str, dict]:
        slot = _tpm.wait_for(gemini_model, _estimate_tokens(messages, max_tokens), settings.llm_tpm_budget)
        result = gemini_client.generate(messages, model=gemini_model, temperature=temperature,
                                        max_tokens=max_tokens, json_mode=json_mode)
        _tpm.settle(slot, result[1].get("total_tokens"))
        return result

    def make_groq(model: str) -> Callable[[], tuple[str, dict]]:
        def run() -> tuple[str, dict]:
            slot = _tpm.wait_for(model, _estimate_tokens(messages, max_tokens), settings.llm_tpm_budget)
            result = groq_client.generate(messages, model=model, temperature=temperature,
                                          max_tokens=max_tokens, json_mode=json_mode)
            _tpm.settle(slot, result[1].get("total_tokens"))
            return result
        return run

    fb_model = settings.groq_fallback_model
    if fb_model and fb_model == groq_model:
        fb_model = ""
    providers = [("groq", groq_model, make_groq(groq_model))]
    if fb_model:
        providers.append(("groq-fallback", fb_model, make_groq(fb_model)))
    providers.append(("gemini", gemini_model, run_gemini))
    if prefer == "gemini":
        providers = [p for p in providers if p[0] == "gemini"] + [p for p in providers if p[0] != "gemini"]

    # Skip providers whose breaker is open — unless that would leave nothing to try.
    live = [p for p in providers if not _breaker.is_open(p[0])]
    order = live or providers

    errors: list[str] = []
    primary_status_error = False
    for name, model, runner in order:
        # The fallback Groq model only helps when the PRIMARY failed with an HTTP status
        # (429 / 404 / 5xx); on connection errors the whole of Groq is unreachable.
        if name == "groq-fallback" and not primary_status_error:
            continue
        try:
            result = _with_retries(name, model, runner)
        except Exception as exc:  # noqa: BLE001 — provider exhausted; try the next one
            _breaker.failure(name)
            if name == "groq":
                primary_status_error = isinstance(exc, groq.APIStatusError)
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        _breaker.success(name)
        return result

    raise RuntimeError("All LLM providers failed after retries. " + " | ".join(errors))


__all__ = ["generate", "ModelTier", "GeminiAPIError"]
