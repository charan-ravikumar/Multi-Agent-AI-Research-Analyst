"""
observability/metrics.py — tiny thread-safe process-wide counters for the monitoring page.

Counters answer "what has this process seen since it started" (retries, errors by HTTP status, circuit-breaker opens,
seconds spent waiting for rate limits, corrective re-asks of structured outputs).  Per-RUN numbers live in the run
record (observability/runs.py); these are the ones that are not tied to one run because they happen inside the shared
LLM client.  They are not persisted: a restart resets them, which is stated on the page.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[str, float] = defaultdict(float)
_started = time.time()


def inc(name: str, amount: float = 1.0, **labels: str | int) -> None:
    key = name + ("{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}" if labels else "")
    with _lock:
        _counters[key] += amount


def snapshot() -> dict[str, float]:
    with _lock:
        return dict(_counters)


def uptime_s() -> float:
    return time.time() - _started


def total(prefix: str) -> float:
    """Sum of every counter whose name (before any labels) equals ``prefix``."""
    with _lock:
        return sum(v for k, v in _counters.items() if k == prefix or k.startswith(prefix + "{"))


def by_label(prefix: str, label: str) -> dict[str, float]:
    """{label value: count} for one counter name, e.g. by_label("llm_error", "status")."""
    out: dict[str, float] = {}
    with _lock:
        for k, v in _counters.items():
            if k.startswith(prefix + "{"):
                labels = dict(p.split("=", 1) for p in k[len(prefix) + 1:-1].split(","))
                if label in labels:
                    out[labels[label]] = out.get(labels[label], 0.0) + v
    return out


def reset() -> None:
    global _started
    with _lock:
        _counters.clear()
        _started = time.time()
