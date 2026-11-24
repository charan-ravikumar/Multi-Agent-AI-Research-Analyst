"""
tools/arxiv.py — arXiv API search (abstracts only; no PDF download).

Etiquette/limits: ≥ ``arxiv_min_interval_seconds`` between requests (process-wide lock),
timeout, response size cap, never raises (returns []).  Results are only *candidates*: the
caller gates them by relevance so off-topic queries (e.g. remote work) don't pull in noise.
"""
from __future__ import annotations

import asyncio
import io
import re
import time
import defusedxml.ElementTree as ET  # arXiv responses are untrusted input
from datetime import datetime

import httpx

from research_analyst.config import settings
from research_analyst.observability.logger import get_logger
from research_analyst.models.search_result import SearchResult
from research_analyst.retrieval.passages import bm25_rank, tokenize

_log = get_logger("arxiv")
_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}
_MAX_BYTES = 600_000
_locks: dict[int, asyncio.Lock] = {}
_last_call = 0.0
_CANDIDATES = 10                    # fetched per request, then re-ranked locally (one request either way)
_CACHE_TTL = 3600.0
_cache: dict[str, tuple[float, list[SearchResult]]] = {}
_pdf_cache: dict[str, tuple[float, str | None]] = {}
_cooldown_until = 0.0          # monotonic time before which arXiv is not called (after a 429)
_COOLDOWN_SECONDS = 120.0


def _get_lock() -> asyncio.Lock:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _locks:
        _locks[loop_id] = asyncio.Lock()
    return _locks[loop_id]


def abs_to_pdf_url(url: str) -> str | None:
    """https://arxiv.org/abs/2401.00001v1 -> https://arxiv.org/pdf/2401.00001v1 (None for anything else)."""
    m = re.match(r"^https://arxiv\.org/abs/([A-Za-z0-9./_-]+)$", url)
    return f"https://arxiv.org/pdf/{m.group(1)}" if m else None


def _extract_pdf_text(data: bytes, max_pages: int) -> str | None:
    """Text of the first ``max_pages`` pages.  Never raises: a malformed PDF is just 'no full text'."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415 — only needed when full text is requested
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages[:max_pages]:
            parts.append(page.extract_text() or "")
        text = "\n\n".join(parts)
        text = re.sub(r"-\n(?=[a-z])", "", text)                 # re-join words hyphenated across lines
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)             # a PDF breaks every printed line: re-flow paragraphs
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()[:120_000] or None
    except Exception as exc:  # noqa: BLE001
        _log.warning(f"pdf text extraction failed: {type(exc).__name__}: {exc}", step="arxiv_pdf_failed")
        return None


async def fetch_pdf_text(abs_url: str, *, client: httpx.AsyncClient | None = None) -> str | None:
    """Download an arXiv paper's PDF and return its text, or None on any failure.

    Guards: URL must be an arxiv.org/abs/... link (so the host is fixed, no SSRF surface); redirects are
    allowed only within arxiv.org; the body is streamed with a hard byte cap; only the first pages are
    parsed; requests share the API's >= 3s spacing and the 429 cooldown."""
    global _last_call, _cooldown_until
    pdf_url = abs_to_pdf_url(abs_url)
    if not pdf_url or time.monotonic() < _cooldown_until:
        return None
    hit = _pdf_cache.get(pdf_url)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL:
        return hit[1]
    own = client is None
    client = client or httpx.AsyncClient(timeout=max(settings.fetch_timeout_seconds, 15.0), follow_redirects=True,
                                         max_redirects=3, headers={"User-Agent": "ResearchAnalystBot/0.1"})
    try:
        async with _get_lock():
            wait = settings.arxiv_min_interval_seconds - (time.monotonic() - _last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_call = time.monotonic()
            async with client.stream("GET", pdf_url) as resp:
                if resp.status_code == 429:
                    _cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
                    return None
                host = resp.url.host or ""
                if resp.status_code != 200 or not (host == "arxiv.org" or host.endswith(".arxiv.org")):
                    return None
                if "pdf" not in resp.headers.get("content-type", "").lower():
                    return None
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > settings.arxiv_pdf_max_bytes:
                        _log.warning("arxiv pdf exceeds size cap; skipped", step="arxiv_pdf_too_big")
                        return None
        text = await asyncio.to_thread(_extract_pdf_text, bytes(buf), settings.arxiv_pdf_max_pages)
        _pdf_cache[pdf_url] = (time.monotonic(), text)
        return text
    except Exception as exc:  # noqa: BLE001
        _log.warning(f"arxiv pdf fetch failed: {type(exc).__name__}: {exc}", step="arxiv_pdf_failed")
        return None
    finally:
        if own:
            await client.aclose()


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
