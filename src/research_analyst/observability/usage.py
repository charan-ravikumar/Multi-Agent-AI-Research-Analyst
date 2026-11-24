"""
core/usage.py — per-run token accounting, optional cost estimate and hard budget.

Every LLM call's ``usage`` dict (from llm/client.py) is recorded under the research
session id.  ``check`` enforces ``settings.token_budget_per_run`` BEFORE a call
(tokens already spent; a single call can overshoot by its own size).
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

from research_analyst.config import settings


class BudgetExceeded(RuntimeError):
    """Raised before an LLM call when the run's token budget is already spent."""


def _prices() -> dict[str, tuple[float, float]]:
    raw = settings.llm_price_per_1m_tokens.strip()
    if not raw:
        return {}
    try:
        return {m: (float(p[0]), float(p[1])) for m, p in json.loads(raw).items()}
    except (ValueError, TypeError, IndexError, AttributeError):
        return {}


def _blank() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def estimate_cost(usage: dict) -> float | None:
    """USD estimate for ONE call from the configured price table, or None if unpriced."""
    p = _prices().get(str(usage.get("model")))
    if not p:
        return None
    return round((usage.get("prompt_tokens") or 0) / 1e6 * p[0] + (usage.get("completion_tokens") or 0) / 1e6 * p[1], 8)


class UsageTracker:
    _MAX_SESSIONS = 200          # bounded: abandoned sessions can't leak memory forever

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._s: OrderedDict[str, dict] = OrderedDict()

    def _entry(self, session_id: str) -> dict:
        e = self._s.get(session_id)
        if e is None:
            e = {"all": _blank(), "by_agent": {}, "by_model": {}, "truncated_calls": 0}
            self._s[session_id] = e
            while len(self._s) > self._MAX_SESSIONS:
                self._s.popitem(last=False)
        return e

    def record(self, session_id: str, agent: str, usage: dict) -> None:
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or (p + c))
        model = str(usage.get("model") or "unknown")
        with self._lock:
            e = self._entry(session_id)
            for bucket in (e["all"], e["by_agent"].setdefault(agent, _blank()),
                           e["by_model"].setdefault(model, _blank())):
                bucket["calls"] += 1
                bucket["prompt_tokens"] += p
                bucket["completion_tokens"] += c
                bucket["total_tokens"] += t
            if usage.get("finish_reason") == "length":
                e["truncated_calls"] += 1

    def total(self, session_id: str) -> int:
        with self._lock:
            e = self._s.get(session_id)
            return e["all"]["total_tokens"] if e else 0

    def check(self, session_id: str) -> None:
        budget = settings.token_budget_per_run
        if budget > 0 and self.total(session_id) >= budget:
            raise BudgetExceeded(
                f"token budget exhausted: {self.total(session_id)} >= {budget} tokens for this run"
            )

    def summary(self, session_id: str) -> dict:
        with self._lock:
            e = self._s.get(session_id)
            if not e:
                return {**_blank(), "by_agent": {}, "by_model": {}, "truncated_calls": 0,
                        "budget": settings.token_budget_per_run, "estimated_cost_usd": None}
            by_model = {m: dict(v) for m, v in e["by_model"].items()}
            out = {**e["all"], "by_agent": {a: dict(v) for a, v in e["by_agent"].items()},
                   "by_model": by_model, "truncated_calls": e["truncated_calls"],
                   "budget": settings.token_budget_per_run}
        prices = _prices()
        if prices:
            cost, unpriced = 0.0, []
            for m, v in by_model.items():
                if m in prices:
                    cost += v["prompt_tokens"] / 1e6 * prices[m][0] + v["completion_tokens"] / 1e6 * prices[m][1]
                else:
                    unpriced.append(m)
            out["estimated_cost_usd"] = round(cost, 6)
            out["unpriced_models"] = unpriced
        else:
            out["estimated_cost_usd"] = None
        return out

    def pop(self, session_id: str) -> None:
        with self._lock:
            self._s.pop(session_id, None)

    def reset(self) -> None:
        with self._lock:
            self._s.clear()


tracker = UsageTracker()
