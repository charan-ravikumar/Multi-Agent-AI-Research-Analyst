"""
core/sanitize.py — defences for untrusted text (web content, LLM output) and user input.

These are *mitigations*, not a guarantee: a sufficiently persuasive snippet can still
bias a model.  What this module does guarantee (and tests assert):

  * untrusted text cannot close our XML-style delimiters or forge our own
    structural tokens ([S:key] source markers, [F1] fact ids, role tags)
  * invisible/control characters are removed and length is bounded
  * user queries are length/shape validated
  * the final Markdown cannot render images (tracking/exfil beacons), raw HTML,
    or non-http(s) links
"""
from __future__ import annotations

import re

from research_analyst.config import settings

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁠-⁤﻿]")
_STRUCT_TAGS = re.compile(
    r"</?\s*(?:source|text|title|url|facts?|evidence|draft|query|system|assistant|user|instructions?|tool|function)\b[^>]*>",
    re.IGNORECASE,
)
_FORGED_MARKERS = re.compile(r"\[\s*S:[0-9a-fA-F]{6}\s*\]|\[\s*F\d+(?:\s*[,;]\s*F\d+)*\s*\]")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_BLOCK = re.compile(r"<\s*(script|style|iframe|object|embed)\b.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"</?\s*[a-zA-Z][^>\n]{0,200}>")
_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*([^)\s]*)[^)]*\)")


def clean_untrusted(text: str | None, max_chars: int | None = None) -> str:
    """Make web/LLM-derived text safe to embed inside our prompt delimiters."""
    if not text:
        return ""
    t = _CONTROL.sub("", text)
    t = _STRUCT_TAGS.sub("", t)
    t = _FORGED_MARKERS.sub("", t)
    return t[:max_chars] if max_chars else t


def safe_url(url: str | None) -> str:
    """Only http(s) URLs without whitespace/angle brackets are passed to prompts."""
    u = (url or "").strip()
    return u if re.match(r"^https?://[^\s<>\"']+$", u) else "(invalid-url)"


def validate_query(query: str | None) -> str:
    """Normalise a user query or raise ValueError with a user-presentable message."""
    q = _CONTROL.sub("", query or "").strip()
    q = re.sub(r"\s+", " ", q)
    if not q:
        raise ValueError("Please enter a research topic.")
    if len(q) > settings.max_query_chars:
        raise ValueError(f"Query too long ({len(q)} chars; max {settings.max_query_chars}).")
    return q


def sanitize_report_markdown(md: str | None) -> str:
    """Strip images, raw HTML and non-http(s) links from LLM-written Markdown."""
    if not md:
        return ""
    t = _HTML_BLOCK.sub("", md)
    t = _MD_IMAGE.sub("", t)
    t = _HTML_TAG.sub("", t)

    def link(m: re.Match) -> str:
        text, url = m.group(1), m.group(2)
        return m.group(0) if re.match(r"^https?://", url, re.IGNORECASE) else text

    return _MD_LINK.sub(link, t)
