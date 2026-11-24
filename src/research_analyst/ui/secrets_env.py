"""
app/secrets_env.py — make Streamlit Community Cloud secrets visible to config.Settings.

Cloud exposes only TOP-LEVEL scalar secrets as environment variables.  Secrets pasted under a
``[section]`` header (a common mistake), or entered in lower case, would otherwise be invisible and
the pipeline would fail with "GROQ_API_KEY is not set".  This flattens nested tables, upper-cases
keys, and exports the ones this app understands — without overriding values that are already set.

Pure function over plain mappings so it is testable without Streamlit.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any

_PREFIXES = (
    "GROQ_", "GEMINI_", "LANGFUSE_", "LLM_", "TOKEN_", "REDIS_", "CRITIC_", "JUDGE_", "QUERY_", "ARXIV_",
    "DEBATE_", "SEARCH_", "FETCH_", "READER_", "RUN_", "MAX_", "LOG_", "PLANNER_", "SOURCE_", "EVIDENCE_",
)
_EXACT = {"ENVIRONMENT", "CACHE_TTL_SECONDS"}


def _walk(m: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    for k, v in m.items():
        if isinstance(v, Mapping):
            yield from _walk(v)
        elif isinstance(v, (str, int, float, bool)):
            yield str(k).upper(), v


def export_secrets_to_env(secrets: Mapping[str, Any], environ: MutableMapping[str, str]) -> list[str]:
    """Copy recognised settings from ``secrets`` into ``environ`` (never overriding non-empty values).
    Returns the exported variable NAMES (never values) for logging."""
    exported: list[str] = []
    for key, value in _walk(secrets):
        if not (key in _EXACT or key.startswith(_PREFIXES)):
            continue
        if not environ.get(key):
            environ[key] = str(value).lower() if isinstance(value, bool) else str(value)
            exported.append(key)
    return exported
