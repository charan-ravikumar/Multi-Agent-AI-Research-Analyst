"""
core/progress.py — in-process progress for the UI's progress bar.

The Streamlit script and the research thread live in the same process, so a small thread-safe
tracker is enough (and avoids tailing the log file, which is also ephemeral on Streamlit Cloud).

Progress is measured in WEIGHTED UNITS, not time: each sub-question branch is 3 units
(search, read, synthesise), the optional review loop is (2*rounds - 1) units, writing is 3.
Real durations differ per unit, so the percentage is an honest *approximation* — it is monotonic,
never reports 100% before the run finishes, and never moves backwards.

Everything is a no-op for sessions that were not ``start``-ed (CLI, evaluation, tests).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

BRANCH_STEPS = 3
WRITING_UNITS = 3
_MAX_SESSIONS = 200


class _Branch:
    def __init__(self, tracker: ProgressTracker, session_id: str):
        self._t, self._sid, self._done, self._closed = tracker, session_id, 0, False

    def step(self, label: str) -> None:
        if self._closed or self._done >= BRANCH_STEPS:
            return
        self._done += 1
        self._t.advance(self._sid, "branches", 1, label)

    def close(self, label: str | None = None) -> None:
        """Count the branch as finished even if it stopped early (failure), exactly once."""
        if self._closed:
            return
        self._closed = True
        remaining = BRANCH_STEPS - self._done
        self._t.advance(self._sid, "branches", remaining, label)
        self._t._branch_finished(self._sid)


class ProgressTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._s: OrderedDict[str, dict] = OrderedDict()

    def start(self, session_id: str, n_branches: int, review_rounds: int = 0) -> None:
        review = (2 * review_rounds - 1) if review_rounds > 0 else 0
        with self._lock:
            self._s[session_id] = {
                "stages": {"branches": [BRANCH_STEPS * n_branches, 0], "review": [review, 0],
                           "writing": [WRITING_UNITS, 0]},
                "n_branches": n_branches, "branches_done": 0, "label": "Starting research…",
                "t0": time.monotonic(), "t_end": None, "finished": False, "cancelled": False, "stream": "",
            }
            while len(self._s) > _MAX_SESSIONS:
                self._s.popitem(last=False)

    def branch(self, session_id: str) -> _Branch:
        return _Branch(self, session_id)

    def advance(self, session_id: str, stage: str, units: int = 1, label: str | None = None) -> None:
        with self._lock:
            s = self._s.get(session_id)
            if not s or stage not in s["stages"]:
                return
            total, done = s["stages"][stage]
            s["stages"][stage][1] = min(total, done + max(0, units))
            if label:
                s["label"] = label

    def finish_stage(self, session_id: str, stage: str, label: str | None = None) -> None:
        with self._lock:
            s = self._s.get(session_id)
            if s and stage in s["stages"]:
                s["stages"][stage][1] = s["stages"][stage][0]
                if label:
                    s["label"] = label

    def stream(self, session_id: str, text: str) -> None:
        """Latest text of the answer being written (cumulative), shown live by the UI."""
        with self._lock:
            s = self._s.get(session_id)
            if s:
                s["stream"] = text

    def set_label(self, session_id: str, label: str) -> None:
        with self._lock:
            s = self._s.get(session_id)
            if s:
                s["label"] = label

    def _branch_finished(self, session_id: str) -> None:
        with self._lock:
            s = self._s.get(session_id)
            if s:
                s["branches_done"] = min(s["n_branches"], s["branches_done"] + 1)

    def finish(self, session_id: str) -> None:
        with self._lock:
            s = self._s.get(session_id)
            if s:
                for st in s["stages"].values():
                    st[1] = st[0]
                s["finished"], s["label"], s["t_end"] = True, "Done", time.monotonic()

    def cancel(self, session_id: str) -> None:
        with self._lock:
            s = self._s.get(session_id)
            if s:
                s["cancelled"], s["label"], s["t_end"] = True, "Stopped by user", time.monotonic()

    def snapshot(self, session_id: str) -> dict | None:
        with self._lock:
            s = self._s.get(session_id)
            if not s:
                return None
            total = sum(v[0] for v in s["stages"].values()) or 1
            done = sum(v[1] for v in s["stages"].values())
            frac = 1.0 if s["finished"] else min(0.99, done / total)
            end = s["t_end"] if s["t_end"] is not None else time.monotonic()
            return {"fraction": frac, "label": s["label"], "elapsed_s": end - s["t0"], "cancelled": s["cancelled"],
                    "branches_done": s["branches_done"], "n_branches": s["n_branches"], "finished": s["finished"],
                    "stream": s.get("stream", "")}

    def pop(self, session_id: str) -> None:
        with self._lock:
            self._s.pop(session_id, None)

    def reset(self) -> None:
        with self._lock:
            self._s.clear()


progress = ProgressTracker()
