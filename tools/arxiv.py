"""
tools/arxiv.py — arXiv API search (abstracts only; no PDF download).

Etiquette/limits: ≥ ``arxiv_min_interval_seconds`` between requests (process-wide lock),
timeout, response size cap, never raises (returns []).  Results are only *candidates*: the
caller gates them by relevance so off-topic queries (e.g. remote work) don't pull in noise.
"""
from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

from config import settings
from core.logger import get_logger
from models.search_result import SearchResult
from tools.passages import bm25_rank, tokenize

_log = get_logger("arxiv")
_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}
_MAX_BYTES = 600_000
_locks: dict[int, asyncio.Lock] = {}
_last_call = 0.0
_CANDIDATES = 10                    # fetched per request, then re-ranked locally (one request either way)
_CACHE_TTL = 3600.0
_cache: dict[str, tuple[float, list[SearchResult]]] = {}
_cooldown_until = 0.0          # monotonic time before which arXiv is not called (after a 429)
_COOLDOWN_SECONDS = 120.0


def _get_lock() -> asyncio.Lock:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _locks:
        _locks[loop_id] = asyncio.Lock()
    return _locks[loop_id]


def build_search_query(text: str, max_terms: int = 5) -> str:
    terms: list[str] = []
    for t in tokenize(text):
        if t not in terms:
            terms.append(t)
    return " AND ".join(f"all:{t}" for t in terms[:max_terms])


def parse_atom(xml_text: str, sub_question: str) -> list[SearchResult]:
    root = ET.fromstring(xml_text)
    out: list[SearchResult] = []
    for e in root.findall("a:entry", _NS):
        title = re.sub(r"\s+", " ", (e.findtext("a:title", "", _NS) or "")).strip()
        summary = re.sub(r"\s+", " ", (e.findtext("a:summary", "", _NS) or "")).strip()
        url = (e.findtext("a:id", "", _NS) or "").strip().replace("http://", "https://")
        if not (title and summary and url.startswith("https://arxiv.org/")):
            continue
        year = (e.findtext("a:published", "", _NS) or "")[:4]
        tag = f"[{year}] " if year.isdigit() else ""         # lets the recency prior in tools/sources see it
        out.append(SearchResult(sub_question=sub_question, url=url, title=title, snippet=f"{tag}{summary[:480]}",
                                full_text=summary[:1500], source_type="arxiv"))
    return out


def rerank(results: list[SearchResult], sub_question: str, keep: int, now_year: int | None = None) -> list[SearchResult]:
    """arXiv's own ordering is loosely relevance-sorted and ignores age.  Re-rank the candidates by BM25 of
    title+abstract against the sub-question, nudged toward recent work, and keep the best ``keep``."""
    if not results:
        return []
    now = now_year or datetime.now().year
    ranks = dict(bm25_rank(sub_question, [f"{r.title} {r.full_text}" for r in results]))
    top = max(ranks.values(), default=1.0) or 1.0

    def score(i: int, r: SearchResult) -> float:
        m = re.match(r"\[(\d{4})\]", r.snippet)
        year = int(m.group(1)) if m else now
        recency = 0.15 if year >= now - 3 else (-0.10 if year <= now - 10 else 0.0)
        return ranks.get(i, 0.0) / top + recency

    order = sorted(range(len(results)), key=lambda i: -score(i, results[i]))
    return [results[i] for i in order[:keep]]


def relevant(sub_question: str, r: SearchResult, min_overlap: int) -> bool:
    return len(set(tokenize(sub_question)) & set(tokenize(f"{r.title} {r.snippet}"))) >= min_overlap


async def search_arxiv(
    query: str,
    *,
    sub_question: str = "",
    max_results: int = 3,
    client: httpx.AsyncClient | None = None,
) -> list[SearchResult]:
    global _last_call, _cooldown_until
    q = build_search_query(query)
    if not q or time.monotonic() < _cooldown_until:
        return []
    cache_key = f"{q}|{max_results}|{sub_question}"
    hit = _cache.get(cache_key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL:
        return list(hit[1])
    own = client is None
    client = client or httpx.AsyncClient(timeout=settings.fetch_timeout_seconds, headers={"User-Agent": "ResearchAnalystBot/0.1"})
    try:
        async with _get_lock():
            wait = settings.arxiv_min_interval_seconds - (time.monotonic() - _last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_call = time.monotonic()
            resp = await client.get(_API, params={"search_query": q, "max_results": max(_CANDIDATES, max_results),
                                                  "sortBy": "relevance", "sortOrder": "descending"})
        if resp.status_code == 429:
            _cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
            _log.warning(f"arxiv rate-limited (429); skipping arXiv for {_COOLDOWN_SECONDS:.0f}s", step="arxiv_rate_limited")
            return []
        if resp.status_code != 200 or len(resp.content) > _MAX_BYTES:
            return []
        results = rerank(parse_atom(resp.text, sub_question or query), sub_question or query, max_results)
        _cache[cache_key] = (time.monotonic(), results)
        return list(results)
    except Exception as exc:  # noqa: BLE001
        _log.warning(f"arxiv search failed: {type(exc).__name__}: {exc}", step="arxiv_failed")
        return []
    finally:
        if own:
            await client.aclose()
