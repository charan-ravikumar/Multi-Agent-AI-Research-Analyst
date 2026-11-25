"""
llm/client.py — INTERNAL orchestration of Gemini calls: model choice, limits, retries.

Do not import this module directly from agents; use the facade:

    from research_analyst import llm
    text, usage = llm.generate(messages, tier=llm.ModelTier.FAST, schema=PlannerOut, return_usage=True)
    turn = llm.tool_turn(system, history, tools)

Gemini is the ONLY provider and there is NO fallback to another provider or model: a call either succeeds on the
chosen model or raises, so a problem is visible instead of being routed somewhere else.

KEY POOL: when two API keys (two free-tier accounts) are configured, every call is routed to the key with the most
headroom (least wait for its per-minute budget, then most requests left today).  A retry prefers the OTHER key.  Each key
has its own RPM/TPM/RPD counters, rate limiter and circuit breaker, because each account has its own quota.  This is the
only place that knows about keys; agents, tracing and the UI see one `llm.generate`.

Policy (one place; the SDK is not asked to retry, so attempts don't multiply):
  * model: ``ModelTier.FAST`` -> settings.gemini_fast_model, ``STRONG`` -> settings.gemini_strong_model, or an explicit
    ``model=``.  Which agents are STRONG is decided in agents/base.py from settings.gemini_strong_agents.
  * free-tier limits (settings.gemini_rate_limits, per key, planned with headroom so the provider's limit is never reached):
    a call WAITS until the chosen key's per-minute request and token budgets have room; once EVERY key's per-DAY budget is
    spent it raises ``DailyQuotaExceeded`` immediately.
  * retryable errors (429, 408, 5xx, timeouts, connection errors) are retried with exponential backoff + jitter,
    honouring Retry-After (capped).  Non-retryable errors (400/401/403/404) raise at once.
  * per-(key, model) circuit breaker: after N consecutive failures that key is skipped for a cooldown; when every key is
    open, calls fail fast.
  * every terminal failure is a ``RuntimeError`` naming the model.
"""
from __future__ import annotations

import itertools
import json
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar, overload
from zoneinfo import ZoneInfo

from google.genai.errors import APIError as GeminiAPIError

from research_analyst.config import settings
from research_analyst.llm import gemini_client
from research_analyst.observability import metrics
from research_analyst.llm.tools import ToolSpec, ToolTurn

logger = logging.getLogger(__name__)
T = TypeVar("T")

_MAX_RETRIES: int = settings.llm_max_retries
_BACKOFF_BASE: float = settings.llm_backoff_base


class ModelTier(str, Enum):
    """FAST -> the high-volume model (most agents); STRONG -> the reasoning model (only where it is needed)."""
    FAST = "fast"
    STRONG = "strong"


class DailyQuotaExceeded(RuntimeError):
    """The model's requests-per-day budget is spent.  Not retried: waiting a minute cannot fix it."""


def model_for(tier: ModelTier, override: str | None = None) -> str:
    if override:
        return override
    return settings.gemini_strong_model if tier is ModelTier.STRONG else settings.gemini_fast_model


# ── limits ───────────────────────────────────────────────────────────────────

def _limits_for(model: str) -> dict[str, int]:
    """Planned per-key limits {"rpm","tpm","rpd"} for ``model``: the documented limits from settings.gemini_rate_limits,
    pulled in by the headroom settings so we stop BEFORE the provider would.  {} = unlimited."""
    try:
        table = json.loads(settings.gemini_rate_limits or "{}")
    except ValueError:
        logger.warning("GEMINI_RATE_LIMITS is not valid JSON; rate limiting disabled")
        return {}
    got = table.get(model) if isinstance(table, dict) else None
    if not isinstance(got, dict):
        return {}
    raw = {k: int(v) for k, v in got.items() if isinstance(v, (int, float))}
    out = dict(raw)
    for k in ("rpm", "tpm"):
        if raw.get(k, 0) > 0:
            out[k] = max(1, int(raw[k] * settings.gemini_limit_headroom))
    if raw.get("rpd", 0) > 0:
        out["rpd"] = max(1, int(raw["rpd"] * (1 - settings.gemini_daily_reserve)))
    return out


class _Quota:
    """Per-day request counter per (key, model), persisted so it survives restarts.  Resets at midnight Pacific
    (Google's quota day).  File: {"date": ..., "counts": {"k1": {model: n}, "k2": {...}}}."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()

    def _load(self) -> dict[str, Any]:
        path = Path(settings.llm_quota_file)
        try:
            data = json.loads(path.read_text("utf-8"))
            if data.get("date") == self._today():
                counts = data.get("counts") or {}
                if any(isinstance(v, int) for v in counts.values()):      # older single-key format {model: n}
                    counts = {"k1": {m: n for m, n in counts.items() if isinstance(n, int)}}
                return {"date": data["date"], "counts": counts}
        except (OSError, ValueError):
            pass
        return {"date": self._today(), "counts": {}}

    def _save(self, data: dict[str, Any]) -> None:
        path = Path(settings.llm_quota_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), "utf-8")
        except OSError as exc:                      # a read-only filesystem must not break a run
            logger.debug("could not persist quota counter: %s", exc)

    def reserve(self, key_id: str, model: str, rpd: int) -> None:
        """Count one request against today's budget for this key, or raise DailyQuotaExceeded when it is already spent."""
        if rpd <= 0:
            return
        with self._lock:
            data = self._load()
            used = int(data["counts"].get(key_id, {}).get(model, 0))
            if used >= rpd:
                raise DailyQuotaExceeded(
                    f"Daily request limit for {model} reached on key {key_id} ({used}/{rpd}); it resets at midnight Pacific time.")
            data["counts"].setdefault(key_id, {})[model] = used + 1
            self._save(data)

    def used(self, model: str, key_id: str | None = None) -> int:
        """Requests today for ``model``: on one key, or summed over every key."""
        with self._lock:
            counts = self._load()["counts"]
            if key_id is not None:
                return int(counts.get(key_id, {}).get(model, 0))
            return sum(int(c.get(model, 0)) for c in counts.values())


_quota = _Quota()


class _RateLimiter:
    """Sliding 60 s windows for requests (RPM) and tokens (TPM) per model.  Blocks the calling worker thread.

    A call *reserves* an estimate, then ``settle`` swaps in the actual token count so over-estimates don't throttle
    later calls for a full minute."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._win: dict[str, deque] = {}

    def wait_for(self, model: str, est_tokens: int, rpm: int, tpm: int) -> list | None:
        if rpm <= 0 and tpm <= 0:
            return None
        est_tokens = min(est_tokens, tpm) if tpm > 0 else est_tokens      # one call bigger than the budget must still run
        while True:
            with self._lock:
                q = self._win.setdefault(model, deque())
                now = time.monotonic()
                while q and now - q[0][0] >= 60.0:
                    q.popleft()
                rpm_ok = rpm <= 0 or len(q) < rpm
                tpm_ok = tpm <= 0 or sum(e[1] for e in q) + est_tokens <= tpm
                if rpm_ok and tpm_ok:
                    entry = [now, est_tokens]
                    q.append(entry)
                    return entry
                wait = 60.0 - (now - q[0][0]) + 0.05
            logger.info("rate limit: waiting %.1fs for %s", wait, model)
            metrics.inc("rate_limit_wait_seconds", max(wait, 0.05), model=model)
            time.sleep(max(wait, 0.05))

    def wait_estimate(self, model: str, est_tokens: int, rpm: int, tpm: int) -> float:
        """Seconds a call would have to wait right now (0 = it fits).  Does not reserve anything."""
        if rpm <= 0 and tpm <= 0:
            return 0.0
        est_tokens = min(est_tokens, tpm) if tpm > 0 else est_tokens
        with self._lock:
            q = self._win.get(model)
            if not q:
                return 0.0
            now = time.monotonic()
            live = [e for e in q if now - e[0] < 60.0]
            if (rpm <= 0 or len(live) < rpm) and (tpm <= 0 or sum(e[1] for e in live) + est_tokens <= tpm):
                return 0.0
            return max(0.0, 60.0 - (now - live[0][0]) + 0.05) if live else 0.0

    @staticmethod
    def settle(entry: list | None, actual_tokens: int | None) -> None:
        if entry is not None and actual_tokens:
            entry[1] = int(actual_tokens)

    def reset(self) -> None:
        with self._lock:
            self._win.clear()


_tpm = _RateLimiter()          # name kept: tests and tools reset it as `client._tpm`


# ── circuit breaker (per model) ──────────────────────────────────────────────

class _Breaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def is_open(self, key: str) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until.get(key, 0.0)

    def success(self, key: str) -> None:
        with self._lock:
            self._fails[key] = 0
            self._open_until.pop(key, None)

    def failure(self, key: str) -> None:
        with self._lock:
            n = self._fails.get(key, 0) + 1
            self._fails[key] = n
            if n >= settings.llm_breaker_threshold:
                self._open_until[key] = time.monotonic() + settings.llm_breaker_cooldown_seconds
                logger.warning("circuit OPEN for %s (%d consecutive failures, cooldown %.0fs)",
                               key, n, settings.llm_breaker_cooldown_seconds)
                metrics.inc("breaker_open", model=key)

    def reset(self) -> None:
        with self._lock:
            self._fails.clear()
            self._open_until.clear()


_breaker = _Breaker()

_EXPECTED_COMPLETION = 600   # reservation cap: most calls finish far below max_tokens


def _estimate_tokens(text_chars: int, max_tokens: int) -> int:
    return text_chars // 4 + min(max_tokens, _EXPECTED_COMPLETION)


# ── retry classification ─────────────────────────────────────────────────────

def _status_of(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, DailyQuotaExceeded):
        return False
    code = _status_of(exc)
    if code is not None:
        return code == 429 or code == 408 or code >= 500
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
    code = _status_of(exc)
    if code is not None and code >= 500:                      # provider overloaded: wait long enough for the spike to pass
        backoff = settings.llm_overload_backoff_seconds * attempt * random.uniform(0.75, 1.0)  # noqa: S311
    else:
        backoff = (_BACKOFF_BASE ** attempt) * random.uniform(0.5, 1.0)  # noqa: S311 — retry jitter, not security
    ra = _retry_after(exc)
    if ra is not None:
        backoff = max(backoff, min(ra, settings.llm_retry_after_cap_seconds))
    return backoff


_rr = itertools.count()          # tie-breaker so equally loaded keys alternate instead of one taking everything


def _choose_key(model: str, est_tokens: int, limits: dict[str, int], avoid: str | None) -> tuple[str, str]:
    """Pick the (key id, secret) with the most headroom for this call.  Raises when no key can serve it."""
    keys = settings.gemini_keys
    if not keys:
        raise RuntimeError("GEMINI_API_KEY is not set")
    rpd = limits.get("rpd", 0)
    exhausted, open_ = 0, 0
    scored = []
    offset = next(_rr)                                   # rotates the starting key so equally loaded keys alternate
    for idx, (key_id, secret) in enumerate(keys):
        if rpd > 0 and _quota.used(model, key_id) >= rpd:
            exhausted += 1
            continue
        if _breaker.is_open(f"{key_id}:{model}"):
            open_ += 1
            continue
        wait = _tpm.wait_estimate(f"{key_id}:{model}", est_tokens, limits.get("rpm", 0), limits.get("tpm", 0))
        remaining = (rpd - _quota.used(model, key_id)) if rpd > 0 else 0
        scored.append(((round(wait, 1), key_id == avoid, -remaining, (idx + offset) % len(keys)), key_id, secret))
    if not scored:
        if exhausted == len(keys):
            raise DailyQuotaExceeded(
                f"Daily request limit for {model} reached on all {len(keys)} key(s); it resets at midnight Pacific time. "
                "Use a different model, or lower GEMINI_STRONG_AGENTS.")
        raise RuntimeError(f"LLM circuit open for {model} on every key: it failed repeatedly; retrying after a short cooldown")
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][2]


def _run(model: str, est_tokens: int, call: Callable[[str], T], tokens_of: Callable[[T], int | None]) -> T:
    """Run ``call(api_key)`` for ``model`` on the best key under the limits, breakers and retry policy.

    Raises RuntimeError when it cannot.  A retry prefers the other key (if any); the result's usage gets ``key_id``."""
    limits = _limits_for(model)
    last: BaseException | None = None
    avoid: str | None = None
    failed_keys: set[str] = set()                # breaker counts one failure per exhausted CALL, not per attempt
    for attempt in range(1, _MAX_RETRIES + 1):
        key_id, secret = _choose_key(model, est_tokens, limits, avoid)     # may raise DailyQuotaExceeded / circuit open
        slot_name = f"{key_id}:{model}"
        try:
            _quota.reserve(key_id, model, limits.get("rpd", 0))
            slot = _tpm.wait_for(slot_name, est_tokens, limits.get("rpm", 0), limits.get("tpm", 0))
            result = call(secret)
            _tpm.settle(slot, tokens_of(result))
            _breaker.success(slot_name)
            metrics.inc("llm_success", model=model, key=key_id)
            _tag_key(result, key_id)
            return result
        except DailyQuotaExceeded:
            metrics.inc("llm_error", model=model, status="daily_quota", key=key_id)
            avoid = key_id                                                # the other key may still have budget
            if len(settings.gemini_keys) > 1 and attempt < _MAX_RETRIES:
                continue
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            last = exc
            metrics.inc("llm_error", model=model, status=_status_of(exc) or type(exc).__name__, key=key_id)
            if not _retryable(exc):
                logger.warning("%s[%s] non-retryable error: %s: %s", model, key_id, type(exc).__name__, exc)
                break
            failed_keys.add(key_id)
            avoid = key_id
            if attempt == _MAX_RETRIES:
                logger.warning("%s[%s] retries exhausted (%d/%d): %s", model, key_id, attempt, _MAX_RETRIES, exc)
                break
            # With a second key the retry can start at once on the other account; waiting only helps a single key.
            wait = 0.0 if len(settings.gemini_keys) > 1 and attempt % 2 == 1 else _delay(attempt, exc)
            metrics.inc("llm_retry", model=model, key=key_id)
            logger.warning("%s[%s] retryable error (attempt %d/%d) — waiting %.1fs: %s", model, key_id, attempt, _MAX_RETRIES, wait, exc)
            if wait:
                time.sleep(wait)
    for key_id in failed_keys:
        _breaker.failure(f"{key_id}:{model}")
    raise RuntimeError(f"LLM call failed on {model} after retries: {type(last).__name__}: {str(last)[:200]}") from last


def _tag_key(result: Any, key_id: str) -> None:
    """Record which key served the call in its usage dict (for the run record); the id is not secret."""
    usage = result[1] if isinstance(result, tuple) and len(result) > 1 else getattr(result, "usage", None)
    if isinstance(usage, dict):
        usage["key_id"] = key_id


# ── public API ───────────────────────────────────────────────────────────────

@overload
def generate(prompt: str | list[dict], *, tier: ModelTier = ..., system: str | None = ..., temperature: float | None = ...,
             max_tokens: int | None = ..., return_usage: Literal[False] = ..., json_mode: bool = ...,
             schema: Any = ..., model: str | None = ..., on_text: Callable[[str], None] | None = ...) -> str: ...
@overload
def generate(prompt: str | list[dict], *, tier: ModelTier = ..., system: str | None = ..., temperature: float | None = ...,
             max_tokens: int | None = ..., return_usage: Literal[True], json_mode: bool = ...,
             schema: Any = ..., model: str | None = ..., on_text: Callable[[str], None] | None = ...) -> tuple[str, dict]: ...
def generate(
    prompt: str | list[dict],
    *,
    tier: ModelTier = ModelTier.FAST,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    return_usage: bool = False,
    json_mode: bool = False,
    schema: Any = None,
    model: str | None = None,
    on_text: Callable[[str], None] | None = None,
) -> str | tuple[str, dict]:
    """Generate text, or schema-constrained JSON when ``schema`` (a Pydantic model class) is given.

    ``on_text`` streams the answer (cumulative text after each chunk) for live display.  ``json_mode`` without a schema asks only for valid JSON.  Returns text, or (text, usage) when ``return_usage``."""
    temp = temperature if temperature is not None else settings.llm_temperature
    max_tok = max_tokens if max_tokens is not None else settings.llm_max_tokens
    messages = _build_messages(prompt, system)
    name = model_for(tier, model)
    est = _estimate_tokens(sum(len(m.get("content", "")) for m in messages), max_tok)
    text, usage = _run(
        name, est,
        lambda key: gemini_client.generate(messages, model=name, temperature=temp, max_tokens=max_tok,
                                           json_mode=json_mode, schema=schema, on_text=on_text, api_key=key),
        lambda r: r[1].get("total_tokens"))
    return (text, usage) if return_usage else text


def tool_turn(
    system: str | None, history: list[Any], tools: list[ToolSpec], *, tier: ModelTier = ModelTier.FAST,
    temperature: float | None = None, max_tokens: int | None = None, model: str | None = None,
) -> ToolTurn:
    """One native function-calling turn.  The model must answer with calls to the declared ``tools``."""
    temp = temperature if temperature is not None else settings.llm_temperature
    max_tok = max_tokens if max_tokens is not None else settings.llm_max_tokens
    name = model_for(tier, model)
    est = _estimate_tokens(len(system or "") + sum(len(str(h)) for h in history) // 2, max_tok)
    return _run(
        name, est,
        lambda key: gemini_client.tool_turn(system, history, tools, model=name, temperature=temp, max_tokens=max_tok, api_key=key),
        lambda t: t.usage.get("total_tokens"))


def _build_messages(prompt: str | list[dict], system: str | None) -> list[dict]:
    if isinstance(prompt, list):
        return prompt
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def quota_status() -> list[dict[str, Any]]:
    """Today's request usage against each configured model's limits, summed over keys, for the monitoring page.

    ``rpd`` is the documented per-key limit times the number of keys; ``keys`` shows the split."""
    try:
        table = json.loads(settings.gemini_rate_limits or "{}")
    except ValueError:
        table = {}
    keys = settings.gemini_keys
    models = list(dict.fromkeys([settings.gemini_fast_model, settings.gemini_strong_model, *table]))
    out = []
    for m in models:
        raw = table.get(m, {}) if isinstance(table.get(m, {}), dict) else {}
        out.append({"model": m, "used": _quota.used(m), "rpd": int(raw.get("rpd", 0)) * max(1, len(keys)),
                    "rpm": int(raw.get("rpm", 0)) * max(1, len(keys)), "tpm": int(raw.get("tpm", 0)) * max(1, len(keys)),
                    "keys": {kid: _quota.used(m, kid) for kid, _ in keys}})
    return out


def daily_requests_used(model: str) -> int:
    return _quota.used(model)


__all__ = ["generate", "tool_turn", "ModelTier", "DailyQuotaExceeded", "GeminiAPIError", "model_for", "daily_requests_used"]
