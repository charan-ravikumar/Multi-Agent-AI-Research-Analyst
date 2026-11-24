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
        if session_id and _propagate is not None:
            stack.enter_context(_propagate(session_id=session_id))
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
def run_span(phase: str, session_id: str, query: str = "") -> Iterator[_Handle]:
    """Root span for one UI/CLI invocation; flushes on exit so short-lived threads don't lose data."""
    with observation(f"run:{phase}", as_type="span", session_id=session_id, input={"query": query}) as h:
        try:
            yield h
        finally:
            flush()


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
