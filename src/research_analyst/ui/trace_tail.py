"""Tail of the shared JSONL trace for the session (collapsed technical log)."""
from __future__ import annotations

import json
import os

from research_analyst.ui.state import LOG_FILE, TRACE_TAIL



def read_trace_tail(session_id: str, n: int = TRACE_TAIL, max_bytes: int = 256_000) -> list[dict]:
    """Return the last *n* log entries for *session_id*.

    Reads only the final ``max_bytes`` of the shared trace file (not the whole file) so the
    cost of a poll does not grow with the log; a partial first line is discarded."""
    if not LOG_FILE.exists():
        return []
    entries: list[dict] = []
    try:
        with open(LOG_FILE, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            chunk = fh.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()
        if size > max_bytes:
            lines = lines[1:]                      # first line is probably cut in half
        for raw in lines:
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if rec.get("session_id") == session_id:
                entries.append(rec)
    except OSError:
        pass
    return entries[-n:]
