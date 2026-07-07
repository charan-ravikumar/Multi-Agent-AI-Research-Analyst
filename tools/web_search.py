"""
tools/web_search.py — DuckDuckGo web search helper

Public API:
    search_web(query, *, sub_question="", max_results=5) -> list[SearchResult]

Called from async agents via asyncio.to_thread (same pattern as llm_generate
in agents/base.py). The underlying DDGS.text() is synchronous.

Uses the `ddgs` package (pip install ddgs).

Field mapping (DDGS -> SearchResult):
    title  -> title
    href   -> url
    body   -> snippet
"""
from __future__ import annotations

import asyncio
import time
from typing import List

from ddgs.exceptions import (
    DDGSException,
    RatelimitException,
    TimeoutException,
)

from core.logger import get_logger
from models import SearchResult

_log = get_logger("web_search")

# Seconds to wait before a single retry on rate-limit or timeout
_RETRY_DELAY = 5.0


def _ddgs_search(query: str, max_results: int) -> list[dict]:
    """
    Synchronous inner call — always run this inside asyncio.to_thread().
    Returns raw DDGS result dicts (keys: title, href, body).
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        return ddgs.text(query, max_results=max_results) or []


async def search_web(
    query: str,
    *,
    sub_question: str = "",
    max_results: int = 5,
) -> List[SearchResult]:
    """
    Async web search using DuckDuckGo.

    Args:
        query: The search query string.
        sub_question: Which research sub-question this search addresses.
                      Defaults to ``query`` if omitted.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of SearchResult objects. Empty list on zero results or
        unrecoverable error — never raises.
    """
    effective_sub_question = sub_question or query

    _log.info(
        f"search_web called: query={query!r} max_results={max_results}",
        step="search_start",
    )

    raw: list[dict] = []

    try:
        raw = await asyncio.to_thread(_ddgs_search, query, max_results)
    except RatelimitException:
        _log.warning(
            f"rate limit hit — sleeping {_RETRY_DELAY}s then retrying: query={query!r}",
            step="search_retry",
        )
        await asyncio.sleep(_RETRY_DELAY)
        try:
            raw = await asyncio.to_thread(_ddgs_search, query, max_results)
        except DDGSException as exc:
            _log.error(
                f"search failed after retry: {exc}",
                step="search_error",
            )
            return []
    except TimeoutException as exc:
        _log.warning(
            f"search timed out — sleeping {_RETRY_DELAY}s then retrying: {exc}",
            step="search_retry",
        )
        await asyncio.sleep(_RETRY_DELAY)
        try:
            raw = await asyncio.to_thread(_ddgs_search, query, max_results)
        except DDGSException as exc2:
            _log.error(
                f"search failed after retry: {exc2}",
                step="search_error",
            )
            return []
    except DDGSException as exc:
        _log.error(
            f"search failed: {exc}",
            step="search_error",
        )
        return []

    if not raw:
        _log.warning(
            f"zero results returned: query={query!r}",
            step="search_done",
        )
        return []

    results = [
        SearchResult(
            sub_question=effective_sub_question,
            url=r.get("href", ""),
            title=r.get("title", ""),
            snippet=r.get("body", ""),
            full_text="",
            source_type="web",
        )
        for r in raw
        if r.get("href", "").startswith("http")  # skip relative/malformed URLs
    ]

    _log.info(
        f"search complete: query={query!r} results={len(results)}",
        step="search_done",
    )

    return results
