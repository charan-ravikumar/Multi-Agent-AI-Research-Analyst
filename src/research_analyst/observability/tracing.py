"""
core/tracing.py — optional Langfuse tracing (Langfuse SDK v3+/OpenTelemetry based).

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are both set.
Disabled = zero overhead and zero network.  Tracing can NEVER break a run: every
SDK call is wrapped, and failures are logged once and swallowed.

Trace shape (one trace per UI/CLI invocation, grouped into a Langfuse *session*
by the research session id):

    run:<phase>                        (core.tracing.run_span at the entry point)
      agent:planner / agent:reader …   (BaseAgent.__call__)
        llm:<agent>                    generation: model, tokens, finish_reason, latency

Context propagates through asyncio tasks and ``asyncio.to_thread`` (both copy
contextvars), so parallel LangGraph branches nest under the run span.
"""
from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from typing import Any
from collections.abc import Iterator

from research_analyst.config import settings

_log = logging.getLogger(__name__)

_client: Any = None
_init_done = False
_propagate: Any = None


def enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _get_client() -> Any:
    global _client, _init_done, _propagate
    if _init_done:
        return _client
    _init_done = True
    if not enabled():
        return None
    try:
        from langfuse import Langfuse, propagate_attributes  # noqa: PLC0415
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_url,
        )
        _propagate = propagate_attributes
        if not _client.auth_check():           # one cheap call; avoids a 401 error on every span batch
            raise RuntimeError(
                f"credentials rejected by {settings.langfuse_url} (keys belong to one region: set LANGFUSE_BASE_URL)"
            )
        _log.info("Langfuse tracing enabled (%s)", settings.langfuse_url)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Langfuse unavailable; tracing disabled: %s", exc)
        _client = None
        _propagate = None
    return _client


def get_client() -> Any:
    """The shared Langfuse client, or None when tracing is disabled/unavailable."""
    return _get_client()


def _truncate(value: Any) -> Any:
    if not settings.langfuse_capture_io:
        return None
    limit = settings.langfuse_max_io_chars
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"…[+{len(value) - limit} chars]"
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    return value


class _Handle:
    """Safe wrapper: .update() never raises, and is a no-op when tracing is off."""

    def __init__(self, obs: Any = None) -> None:
        self._obs = obs

    def update(self, **kw: Any) -> None:
        if self._obs is None:
            return
        if "input" in kw:
            kw["input"] = _truncate(kw["input"])
        if "output" in kw:
            kw["output"] = _truncate(kw["output"])
        try:
            self._obs.update(**kw)
        except Exception as exc:  # noqa: BLE001
            _log.debug("langfuse update failed: %s", exc)


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    session_id: str | None = None,
    input: Any = None,  # noqa: A002
    tags: list[str] | None = None,
    **kw: Any,
) -> Iterator[_Handle]:
    """Open a nested observation; yields a _Handle.  Exceptions from the body are
    recorded on the observation (level=ERROR) and re-raised unchanged."""
    client = _get_client()
    if client is None:
        yield _Handle()
        return
    stack = ExitStack()
    handle = _Handle()
    try:
        if (session_id or tags) and _propagate is not None:
            attrs: dict[str, Any] = {}
            if session_id:
                attrs["session_id"] = session_id
            if tags:
                attrs["tags"] = tags               # filterable in Langfuse (model, review on/off, env)
            stack.enter_context(_propagate(**attrs))
        obs = stack.enter_context(
            client.start_as_current_observation(
                name=name, as_type=as_type, input=_truncate(input), **kw
            )
        )
        handle = _Handle(obs)
    except Exception as exc:  # noqa: BLE001
        _log.warning("langfuse observation '%s' failed to start: %s", name, exc)
        stack.close()
        stack = ExitStack()
    try:
        yield handle
    except BaseException as exc:
        handle.update(level="ERROR", status_message=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
    finally:
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001
            _log.debug("langfuse close failed: %s", exc)


@contextmanager
def run_span(phase: str, session_id: str, query: str = "", tags: list[str] | None = None) -> Iterator[_Handle]:
    """Root span for one UI/CLI invocation; flushes on exit so short-lived threads don't lose data.

    Tagged with the environment and strong model so runs can be filtered/compared in Langfuse; callers add
    their own (e.g. "review:on")."""
    base = [f"env:{settings.environment}", f"model:{settings.gemini_strong_model}"]
    with observation(f"run:{phase}", as_type="span", session_id=session_id, input={"query": query},
                     tags=[*base, *(tags or [])]) as h:
        try:
            yield h
        finally:
            flush()


def run_scores(state: dict) -> dict[str, float]:
    """Numeric quality/cost metrics of a finished run (pure function, so it is testable and reusable)."""
    cs = state.get("citation_stats") or {}
    us = state.get("usage_summary") or {}
    gaps = state.get("gap_items") or []
    out: dict[str, float] = {
        "report_words": float(len((state.get("final_report") or "").split())),
        "branch_failures": float(sum(1 for r in state.get("routing_history", []) if str(r).startswith("branch_failed"))),
        "coverage_warnings": float(sum(1 for g in gaps if g.get("kind") == "coverage")),
        "open_questions": float(sum(1 for g in gaps if g.get("kind") == "open_question")),
        "review_loop_forced": 1.0 if state.get("debate_forced") else 0.0,
    }
    for src, key in [(cs, "cited_ratio"), (cs, "uncited"), (cs, "weakly_supported"), (cs, "sources_cited"),
                     (us, "total_tokens"), (us, "calls"), (us, "estimated_cost_usd")]:
        v = src.get(key)
        if isinstance(v, (int, float)):
            out["llm_calls" if key == "calls" else key] = float(v)
    return out


def record_scores(state: dict | None) -> None:
    """Attach the run's metrics to the CURRENT Langfuse trace as scores (call inside ``run_span``)."""
    client = _get_client()
    if client is None or not state:
        return
    try:
        trace_id = client.get_current_trace_id()
        for name, value in run_scores(state).items():
            client.create_score(name=name, value=value, trace_id=trace_id, data_type="NUMERIC")
    except Exception as exc:  # noqa: BLE001 — scoring must never affect a run
        _log.debug("langfuse score failed: %s", exc)


def flush() -> None:
    client = _client
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001
        _log.debug("langfuse flush failed: %s", exc)


def reset_for_tests() -> None:
    global _client, _init_done, _propagate
    _client, _init_done, _propagate = None, False, None
