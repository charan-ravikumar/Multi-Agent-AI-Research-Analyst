"""
grounding/verify.py — "click a claim, see its evidence".

Given the final report and the evidence the pipeline gathered, produce for every cited sentence the
passage(s) of each cited source that best match it, with the shared terms highlighted.

What this is — and is not: it shows the reader WHERE a claim came from so they can check it in seconds.
The LEXICAL support label (strong / partial / weak) is term overlap — it is only the fallback now.  The primary
verdict comes from an LLM judge (grounding/judge.py) that reads the claim and the cited passages and decides
supported / partial / unsupported.  BM25 here only chooses WHICH passage to show and send to the judge.

Security: passages come from web pages.  Every text segment is HTML-escaped; the only markup this module
ever emits is its own ``<mark>`` tags (see ``highlight``).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from research_analyst.retrieval.passages import bm25_rank, split_passages, tokenize

_SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z\"'(\[])")
_NUM = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[A-Za-z0-9]{3,}")
_MIN_CHARS = 40
STRONG, PARTIAL = 0.5, 0.25


@dataclass(frozen=True)
class Claim:
    id: int
    text: str            # the sentence without citation markers
    refs: tuple[int, ...]


def _get(obj: Any, key: str, default: Any = "") -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def extract_claims(report: str, max_claims: int = 40) -> list[Claim]:
    """Cited sentences of the report body (everything before the References section)."""
    body = re.split(r"(?im)^#{1,3}\s*references\b", report or "", maxsplit=1)[0]
    claims: list[Claim] = []
    seen: set[str] = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        for sent in _SENT_SPLIT.split(line):
            refs = tuple(dict.fromkeys(int(n) for n in _NUM.findall(sent)))
            text = re.sub(r"\s+", " ", re.sub(r"\s*\[\d+\]", "", sent)).strip(" *_")
            if refs and len(text) >= _MIN_CHARS and text not in seen:
                seen.add(text)
                claims.append(Claim(len(claims) + 1, text, refs))
                if len(claims) >= max_claims:
                    return claims
    return claims


def build_evidence_index(search_results: list[Any], facts: list[Any], references: list[dict],
                         max_chars: int = 1500) -> dict[str, dict]:
    """{ref number (as str, JSON-safe): {title, url, text, facts}} for the sources actually cited."""
    text_by_url: dict[str, str] = {}
    for r in search_results:
        url, full, snip = _get(r, "url"), _get(r, "full_text"), _get(r, "snippet")
        best = (full or snip or "").strip()
        if len(best) > len(text_by_url.get(url, "")):
            text_by_url[url] = best
    facts_by_url: dict[str, list[str]] = {}
    for f in facts:
        facts_by_url.setdefault(_get(f, "source_url"), []).append(str(_get(f, "content")))
    return {
        str(ref["n"]): {"title": ref["title"], "url": ref["url"],
                        "text": text_by_url.get(ref["url"], "")[:max_chars],
                        "facts": facts_by_url.get(ref["url"], [])[:6]}
        for ref in references
    }


def highlight(text: str, terms: set[str]) -> str:
    """HTML for ``text`` with words in ``terms`` wrapped in <mark>.  Every segment is escaped."""
    out: list[str] = []
    pos = 0
    for m in _WORD.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        word = html.escape(m.group(0))
        out.append(f"<mark>{word}</mark>" if m.group(0).lower() in terms else word)
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _label(score: float) -> str:
    return "strong" if score >= STRONG else "partial" if score >= PARTIAL else "weak"


def best_support(claim_text: str, ev: dict) -> tuple[str, str]:
    """The passage and the extracted fact of one cited source that best match the claim (BM25).  Retrieval, not judgement."""
    passages = split_passages(ev.get("text", ""), target=120, max_len=520) or ([ev["text"]] if ev.get("text") else [])
    best_passage = ""
    if passages:
        ranked = bm25_rank(claim_text, passages)
        best_passage = passages[ranked[0][0]] if ranked else passages[0]
    facts = ev.get("facts", [])
    best_fact = ""
    if facts:
        ranked = bm25_rank(claim_text, facts)
        best_fact = facts[ranked[0][0]] if ranked else facts[0]
    return best_passage, best_fact


def support_for(claim: Claim, evidence: dict[str, dict]) -> list[dict]:
    """For each source cited by ``claim``: the best passage + best extracted fact, a lexical support label."""
    claim_tokens = set(tokenize(claim.text))
    out = []
    for n in claim.refs:
        ev = evidence.get(str(n))                                  # keys are strings (JSON round-trips turn ints into str)
        if not ev:
            out.append({"ref": n, "title": "(source not found)", "url": "", "label": "weak", "score": 0.0,
                        "passage_html": "", "fact_html": ""})
            continue
        best_passage, best_fact = best_support(claim.text, ev)
        support_tokens = set(tokenize(best_passage)) | set(tokenize(best_fact))
        shared = claim_tokens & support_tokens
        score = round(len(shared) / len(claim_tokens), 3) if claim_tokens else 0.0
        out.append({
            "ref": n, "title": ev.get("title", ""), "url": ev.get("url", ""),
            "label": _label(score), "score": score,
            "passage_html": highlight(best_passage, shared) if best_passage else "",
            "fact_html": highlight(best_fact, shared) if best_fact else "",
        })
    return out


def verify_payload(report: str, search_results: list[Any], facts: list[Any], references: list[dict],
                   claim_verdicts: dict | None = None) -> dict:
    """Everything the verify panel needs, JSON-serialisable (used by the live results page and the gallery).

    ``claim_verdicts`` is the LLM judge's output (grounding/judge.py); when absent the panel falls back to the
    lexical word-overlap label and says so.
    """
    return {"claims": [{"id": c.id, "text": c.text, "refs": list(c.refs)} for c in extract_claims(report)],
            "evidence": build_evidence_index(search_results, facts, references),
            "verdicts": claim_verdicts or {}}
