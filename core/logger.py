"""
core/logger.py — structured JSON-lines logger

Every log record is emitted as a single JSON object with these guaranteed fields:

    timestamp   ISO-8601 UTC string
    level       DEBUG | INFO | WARNING | ERROR | CRITICAL
    agent       name of the agent emitting the record (or "system")
    step        human-readable name for the operation (e.g. "llm_call", "scrape")
    session_id  research session identifier
    iteration   reflection-loop counter (0-based)
    latency_s   wall-clock seconds for the operation (None when not timed)
    tokens      dict with prompt/completion/total keys (None when not an LLM call)
    message     free-text description
    extra       any additional key/value pairs passed by the caller

Usage
-----
    from core.logger import get_logger

    log = get_logger("planner")

    # plain info
    log.info("sub-questions generated", step="plan", session_id=sid, iteration=0)

    # timed LLM call — pass the usage dict from llm/client.py
    log.info(
        "LLM call complete",
        step="llm_call",
        session_id=sid,
        iteration=1,
        latency_s=0.83,
        tokens={"prompt_tokens": 312, "completion_tokens": 128, "total_tokens": 440},
    )

Forward-compatibility
---------------------
The `_build_record` function returns a plain dict, making it trivial to forward
records to Langfuse (as observations) or OpenTelemetry (as span events) later
without changing any call-sites.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings

# ── JSON formatter ────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Render every LogRecord as a single-line JSON object."""

    # Fields injected via `extra=` that we promote to top-level keys
    _TOP_LEVEL = {"agent", "step", "session_id", "iteration", "latency_s", "tokens"}

    def format(self, record: logging.LogRecord) -> str:
        record_dict = _build_record(record, self._TOP_LEVEL)
        return json.dumps(record_dict, default=str, ensure_ascii=False)


def _build_record(record: logging.LogRecord, top_level_keys: set[str]) -> dict:
    """
    Build the canonical log dict from a LogRecord.
    Separates well-known fields from arbitrary extra kwargs.
    """
    # Pull well-known extras directly from the record object (set via extra={})
    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    for key, value in record.__dict__.items():
        if key in top_level_keys:
            known[key] = value
        elif key not in _LOGGING_INTERNAL_ATTRS:
            extra[key] = value

    return {
        "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
        "level": record.levelname,
        "agent": known.get("agent", "system"),
        "step": known.get("step", ""),
        "session_id": known.get("session_id", ""),
        "iteration": known.get("iteration", 0),
        "latency_s": known.get("latency_s"),
        "tokens": known.get("tokens"),
        "message": record.getMessage(),
        "logger": record.name,
        **({"extra": extra} if extra else {}),
        **({"exc_info": _format_exc(record)} if record.exc_info else {}),
    }


def _format_exc(record: logging.LogRecord) -> str | None:
    if record.exc_info:
        return logging.Formatter().formatException(record.exc_info)
    return None


# Attributes that belong to the LogRecord internals — never promote these
_LOGGING_INTERNAL_ATTRS: frozenset[str] = frozenset({
    "name", "msg", "args", "created", "filename", "funcName", "levelname",
    "levelno", "lineno", "module", "msecs", "message", "pathname", "process",
    "processName", "relativeCreated", "stack_info", "thread", "threadName",
    "exc_info", "exc_text", "taskName",
    # well-known extras we already promote
    "agent", "step", "session_id", "iteration", "latency_s", "tokens",
})


# ── handler setup ─────────────────────────────────────────────────────────────

def _build_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = []

    # 1. stderr (always active — plain text in dev, JSON in non-dev)
    stream_handler = logging.StreamHandler(sys.stderr)
    if settings.environment == "development":
        # Human-readable in dev so the terminal isn't flooded with JSON
        stream_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    else:
        stream_handler.setFormatter(_JSONFormatter())
    handlers.append(stream_handler)

    # 2. JSON-lines file (always active)
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(_JSONFormatter())
    handlers.append(file_handler)

    return handlers


_handlers = _build_handlers()
_root_configured = False


def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    for h in _handlers:
        root.addHandler(h)
    _root_configured = True


# ── public API ────────────────────────────────────────────────────────────────

def get_logger(agent_name: str) -> "AgentLogger":
    """
    Return a logger bound to *agent_name*.

    All records emitted through this logger automatically include
    ``agent=agent_name`` so you never have to pass it at each call site.

        log = get_logger("planner")
        log.info("done", step="plan", session_id=sid, iteration=0)
    """
    _configure_root()
    return AgentLogger(agent_name)


class AgentLogger:
    """
    Thin wrapper around :class:`logging.Logger` that:
    - pre-fills ``agent`` in every record
    - enforces the standard extra-field contract
    - exposes a convenience :meth:`log_call` for LLM / tool timing
    """

    def __init__(self, agent_name: str) -> None:
        self._agent = agent_name
        self._logger = logging.getLogger(f"agents.{agent_name}")

    def _extra(self, kwargs: dict) -> dict:
        """Merge agent name into caller-supplied extras."""
        base = {"agent": self._agent}
        base.update(kwargs)
        return base

    # ── standard levels ───────────────────────────────────────────────────────

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(message, extra=self._extra(kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, extra=self._extra(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(message, extra=self._extra(kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.error(message, extra=self._extra(kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        self._logger.critical(message, extra=self._extra(kwargs))

    # ── convenience helper ────────────────────────────────────────────────────

    def log_call(
        self,
        step: str,
        *,
        session_id: str,
        iteration: int = 0,
        latency_s: float | None = None,
        tokens: dict | None = None,
        message: str = "",
        level: str = "INFO",
        **extra: Any,
    ) -> None:
        """
        Emit a structured record for a single agent step (LLM call, tool call,
        scrape, etc.).

        Parameters
        ----------
        step        : operation name, e.g. "llm_call", "ddg_search", "scrape"
        session_id  : research session identifier
        iteration   : current reflection-loop counter
        latency_s   : wall-clock duration in seconds
        tokens      : dict from llm/client.py usage — keys:
                      prompt_tokens, completion_tokens, total_tokens, provider, model
        message     : free-text description
        level       : log level string (default INFO)
        **extra     : any additional fields stored under the "extra" key
        """
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(
            message or step,
            extra=self._extra(
                dict(
                    step=step,
                    session_id=session_id,
                    iteration=iteration,
                    latency_s=latency_s,
                    tokens=tokens,
                    **extra,
                )
            ),
        )
