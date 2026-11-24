"""
core/json_parse.py — one tolerant JSON extractor for every LLM stage.

Replaces the per-agent ``_strip_fences`` + ``json.loads`` copies.  Handles the
failure modes actually seen in logs/agent_traces.jsonl:

  * literal newlines inside JSON strings   ("Invalid control character")  -> strict=False
  * trailing text after the JSON value     ("Extra data")                  -> raw_decode
  * ```json fences / leading prose                                         -> scan for first { or [
  * ``None`` / empty provider text                                         -> ValueError

Raises ``ValueError`` (``json.JSONDecodeError`` is a subclass) so existing
retry-on-parse-error handlers keep working.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_DECODER = json.JSONDecoder(strict=False)


def parse_json(raw: str | None) -> Any:
    if raw is None or not raw.strip():
        raise ValueError("empty LLM response (None or blank)")
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as first:
        for i, ch in enumerate(text):
            if ch in "{[":
                try:
                    obj, _end = _DECODER.raw_decode(text[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
        raise first
