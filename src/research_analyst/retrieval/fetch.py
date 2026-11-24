"""
tools/fetch.py — bounded, SSRF-guarded page fetching + main-text extraction.

Search snippets are 1-2 sentences; this lets the Reader work from real page
text.  Design constraints (all enforced here, all tested offline):

  * http/https only; every hop (incl. redirects, max 3) is validated
  * host must resolve ONLY to public IPs (blocks loopback / RFC1918 / link-local /
    metadata endpoints) — search results are untrusted input
  * hard timeout, hard byte cap (streamed), HTML content types only
  * bounded concurrency
  * never raises: failures return None and the caller keeps the snippet

Known limit: validation resolves DNS separately from the connect, so a DNS
rebinding attacker could still race it; pinning the resolved IP would need a
custom transport.  Acceptable for a research tool without credentials in scope.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import urljoin, urlparse

import httpx

from research_analyst.config import settings
from research_analyst.observability.logger import get_logger

_log = get_logger("fetch")

_MAX_REDIRECTS = 3
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_HEADERS = {"User-Agent": "ResearchAnalystBot/0.1 (+research tool)", "Accept": "text/html,application/xhtml+xml"}

Resolver = Callable[[str], Awaitable[list[str]]]


async def _default_resolver(host: str) -> list[str]:
    infos = await asyncio.to_thread(socket.getaddrinfo, host, None, proto=socket.IPPROTO_TCP)
    return sorted({str(i[4][0]) for i in infos})


def _is_public(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    return not (a.is_private or a.is_loopback or a.is_link_local or a.is_multicast
                or a.is_reserved or a.is_unspecified)


async def is_safe_url(url: str, resolver: Resolver | None = None) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        ips = await (resolver or _default_resolver)(p.hostname)
    except Exception:  # noqa: BLE001 — unresolvable == unsafe
        return False
    return bool(ips) and all(_is_public(ip) for ip in ips)


def extract_main_text(html: str) -> str | None:
    import trafilatura  # noqa: PLC0415 — heavy import, only needed when fetching
    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
    return text.strip() if text and text.strip() else None


async def fetch_text(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    resolver: Resolver | None = None,
) -> str | None:
    """Fetch ``url`` and return its main text, or None on any failure/refusal."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=settings.fetch_timeout_seconds, headers=_HEADERS, follow_redirects=False)
    try:
        current = url
        for _hop in range(_MAX_REDIRECTS + 1):
            if not await is_safe_url(current, resolver):
                _log.warning(f"refused unsafe/unresolvable url: {current}", step="fetch_refused")
                return None
            async with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    loc = resp.headers.get("location")
                    if not loc:
                        return None
                    current = urljoin(current, loc)
                    continue
                if resp.status_code != 200:
                    return None
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in _HTML_TYPES:
                    return None
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > settings.fetch_max_bytes:
                        break                                   # truncate; extract from what we have
                html = bytes(buf[: settings.fetch_max_bytes]).decode(resp.encoding or "utf-8", errors="replace")
            return await asyncio.to_thread(extract_main_text, html)
        return None                                              # too many redirects
    except Exception as exc:  # noqa: BLE001 — network failure must never fail a branch
        _log.warning(f"fetch failed: {url}: {type(exc).__name__}: {exc}", step="fetch_failed")
        return None
    finally:
        if own:
            await client.aclose()


async def fetch_many(
    urls: Iterable[str],
    *,
    client: httpx.AsyncClient | None = None,
    resolver: Resolver | None = None,
) -> dict[str, str]:
    """Fetch several URLs with bounded concurrency; returns {url: text} for successes only."""
    urls = list(dict.fromkeys(urls))
    sem = asyncio.Semaphore(settings.fetch_max_concurrency)
    own = client is None
    client = client or httpx.AsyncClient(timeout=settings.fetch_timeout_seconds, headers=_HEADERS, follow_redirects=False)

    async def one(u: str) -> tuple[str, str | None]:
        async with sem:
            return u, await fetch_text(u, client=client, resolver=resolver)

    try:
        pairs = await asyncio.gather(*(one(u) for u in urls))
    finally:
        if own:
            await client.aclose()
    return {u: t for u, t in pairs if t}
