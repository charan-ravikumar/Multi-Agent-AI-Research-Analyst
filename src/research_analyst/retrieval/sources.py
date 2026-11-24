"""
tools/sources.py — heuristic source-quality scoring, rank fusion and domain diversification.

The scores are a transparent PRIOR (domain tier + a small recency nudge), not a truth detector:
a government page can be wrong and a blog can be right.  They exist to (a) order candidates
before the expensive fetch/Reader stages, and (b) stop one domain from supplying all evidence.
"""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urlparse

_HIGH = (
    "nih.gov", "nlm.nih.gov", "who.int", "europa.eu", "oecd.org", "worldbank.org", "imf.org", "un.org",
    "nature.com", "science.org", "sciencedirect.com", "springer.com", "wiley.com", "thelancet.com", "nejm.org",
    "bmj.com", "jamanetwork.com", "cell.com", "plos.org", "pnas.org", "ieee.org", "acm.org", "arxiv.org",
    "nasa.gov", "iea.org", "annualreviews.org", "jstor.org", "cambridge.org", "oup.com", "tandfonline.com", "sagepub.com", "fda.gov", "cdc.gov", "nist.gov", "pubmed.ncbi.nlm.nih.gov", "biorxiv.org", "medrxiv.org",
)
_GOOD = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com", "wsj.com", "ft.com", "economist.com",
    "bloomberg.com", "technologyreview.com", "statnews.com", "wired.com", "arstechnica.com", "hbr.org",
    "mckinsey.com", "brookings.edu", "britannica.com", "theguardian.com", "washingtonpost.com", "scientificamerican.com",
    "frontiersin.org", "mdpi.com", "quantamagazine.org", "medlineplus.gov", "mayoclinic.org", "clevelandclinic.org",
)
_MID_LOW = ("wikipedia.org", "github.com", "medium.com", "substack.com", "youtube.com", "reddit.com")
_LOW = (
    "pinterest.com", "quora.com", "facebook.com", "instagram.com", "tiktok.com", "ehow.com", "answers.com",
    "slideshare.net", "scribd.com", "twitter.com", "x.com", "linkedin.com",
)
# Platforms that merely sit on an academic-looking TLD or host user uploads: never "high".
_NOT_HIGH = ("academia.edu", "researchgate.net", "sciencedirect.com.cn")
_HIGH_SUFFIX = (".gov", ".edu", ".ac.uk", ".gov.uk", ".edu.au", ".ac.in", ".gov.in")
_YEAR = re.compile(r"\b(20[0-3]\d|19[89]\d)\b")


def host_of(url: str) -> str:
    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def _matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def domain_of(url: str) -> str:
    """Approximate registered domain (last two labels; three for common 2-part public suffixes)."""
    parts = host_of(url).split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in {"co.uk", "ac.uk", "gov.uk", "com.au", "edu.au", "co.in", "ac.in", "gov.in"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host_of(url)


def score_source(url: str, title: str = "", snippet: str = "", now_year: int | None = None) -> float:
    """0..1 prior: domain tier, plus +0.05 for recent (<= 2y) and -0.10 for stale (>= 10y) year mentions."""
    host = host_of(url)
    if _matches(host, _NOT_HIGH):
        base = 0.5
    elif _matches(host, _HIGH) or host.endswith(_HIGH_SUFFIX):
        base = 0.95
    elif _matches(host, _GOOD):
        base = 0.75
    elif _matches(host, _LOW):
        base = 0.2
    elif _matches(host, _MID_LOW):
        base = 0.45
    else:
        base = 0.5
    years = [int(y) for y in _YEAR.findall(f"{url} {title} {snippet}")]
    if years:
        now = now_year or datetime.now().year
        newest = max(years)
        if newest >= now - 2:
            base += 0.05
        elif newest <= now - 10:
            base -= 0.10
    return round(min(1.0, max(0.0, base)), 3)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal-rank fusion over several ranked URL lists -> {url: score}."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, url in enumerate(ranking, 1):
            scores[url] = scores.get(url, 0.0) + 1.0 / (k + rank)
    return scores


def rank_and_diversify(results: list, fused: dict[str, float], limit: int, quality_weight: float = 0.5,
                       max_per_domain: int = 2) -> list:
    """Order by (normalised fused rank) blended with quality; cap results per domain; keep ``limit``.
    Results are SearchResult-like (``url``, ``title``, ``snippet``) and get ``quality`` set."""
    if not results:
        return []
    top = max(fused.values()) if fused else 1.0
    scored = []
    for r in results:
        q = score_source(r.url, r.title, r.snippet)
        rel = (fused.get(r.url, 0.0) / top) if top else 0.0
        scored.append((rel * (1 - quality_weight) + q * quality_weight, q, r))
    scored.sort(key=lambda x: -x[0])
    kept: list = []
    overflow: list = []
    per_domain: dict[str, int] = {}
    for _, q, r in scored:
        d = domain_of(r.url)
        if per_domain.get(d, 0) < max_per_domain:
            per_domain[d] = per_domain.get(d, 0) + 1
            kept.append(r.model_copy(update={"quality": q}))
        else:
            overflow.append(r.model_copy(update={"quality": q}))
    return (kept + overflow)[:limit]            # overflow only fills if diversity left room
