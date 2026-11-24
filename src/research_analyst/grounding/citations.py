"""
core/citations.py — claim-level citation plumbing (pure functions, no LLM).

Lifecycle of a citation marker:
  1. Synthesizer is shown facts as F1..Fn and cites them inline: "... [F2][F4]"
  2. ``convert_fact_markers`` rewrites those to stable source markers "[S:ab12cd]"
     (hash of the URL); ids that don't exist are dropped.
  3. Critic / reviser see "[S:ab12cd]" plus an evidence list using the same keys.
  4. Writer calls ``number_citations``: markers become "[1]", "[2]" ... in order of
     first appearance, and the References list is BUILT BY CODE from the sources
     that were actually cited — the LLM never writes reference entries.
  5. ``verify_numbered`` strips any "[n]" the Writer invented.
  6. ``analyze`` reports how much of the text is cited and how much of the cited
     text is lexically supported by the cited facts (a cheap, deterministic check —
     NOT a semantic entailment proof).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any
from collections.abc import Iterable, Sequence

from research_analyst.safety.sanitize import clean_untrusted
from research_analyst.retrieval.passages import tokenize

_FACT_GROUP = re.compile(r"\[\s*(F\d+(?:\s*[,;]\s*F\d+)*)\s*\]")
_SRC = re.compile(r"\[S:([0-9a-f]{6})\]")
_NUM = re.compile(r"\[(\d+)\]")
_SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z\"'(])")
_MIN_SENTENCE_CHARS = 40
_MIN_SHARED_TOKENS = 2


_OPEN, _CLOSE = "【〔［", "】〕］"
_DAGGER = re.compile(r"\[(\d+)†[^\]]*\]")
_LIST = re.compile(r"\[(\d+(?:\s*[,;，、]\s*\d+)+)\]")
_RANGE = re.compile(r"\[(\d+)\s*[-–—]\s*(\d+)\]")


def normalize_citation_brackets(text: str, max_ref: int | None = None) -> str:
    """Models do not reliably emit ASCII ``[1]``.  Seen in a real run: gpt-oss wrote ``【1】`` for every citation,
    which silently erased all of them downstream.  Normalise at every LLM boundary:

      【1】 〔1〕 ［1］     -> [1]
      [1†source] [3†L5-L9] -> [1] [3]            (browsing-style suffixes)
      [1, 2] [1;2]         -> [1][2]
      [1-3]                -> [1][2][3]          (small ranges only)

    ``max_ref`` limits list/range expansion to plausible reference numbers, so ``[2020, 2021]`` is untouched.
    Markers like [S:ab12cd] / [F2] are never altered."""
    for o in _OPEN:
        text = text.replace(o, "[")
    for c in _CLOSE:
        text = text.replace(c, "]")
    text = _DAGGER.sub(r"[\1]", text)

    def expand_list(m: re.Match) -> str:
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        if max_ref is not None and any(n < 1 or n > max_ref for n in nums):
            return m.group(0)
        return "".join(f"[{n}]" for n in nums)

    def expand_range(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if not (1 <= a < b <= a + 8) or (max_ref is not None and b > max_ref):
            return m.group(0)
        return "".join(f"[{n}]" for n in range(a, b + 1))

    return _RANGE.sub(expand_range, _LIST.sub(expand_list, text))


_CLAIM_SECTIONS = ("executive summary", "key findings", "detailed analysis")


def report_citation_stats(report_body: str) -> dict:
    """Citation coverage of the FINAL report text (the pipeline's other stats describe the draft).

    Only the *claim* sections are measured — Executive Summary, Key Findings, Detailed Analysis.  The
    Contradictions / Knowledge Gaps sections (and their "Open question:" lines) describe what is NOT known and
    are legitimately uncited, so counting them would understate how well the report's claims are sourced.
    A body with no recognised headings is measured whole."""
    has_headings = bool(re.search(r"^#{1,3}\s", report_body, flags=re.M))
    in_claims = not has_headings
    total = cited = 0
    for line in report_body.splitlines():
        line = line.strip()
        if line.startswith("#"):
            in_claims = any(s in line.lower() for s in _CLAIM_SECTIONS)
            continue
        if not line or not in_claims:
            continue
        for sent in _SENT_SPLIT.split(re.sub(r"^[-*•]\s+", "", line)):
            if len(sent.strip()) >= _MIN_SENTENCE_CHARS:
                total += 1
                cited += 1 if _NUM.search(sent) else 0
    return {"report_claim_sentences": total, "report_cited": cited,
            "report_cited_ratio": round(cited / total, 3) if total else 0.0}


def source_key(url: str) -> str:
    return hashlib.sha1(url.strip().encode("utf-8"), usedforsecurity=False).hexdigest()[:6]


def marker(url: str) -> str:
    return f"[S:{source_key(url)}]"


def _dedupe_adjacent(text: str) -> str:
    return re.sub(r"(\[S:[0-9a-f]{6}\])(?:\s*\1)+", r"\1", text)


def convert_fact_markers(text: str, id_to_url: dict[str, str]) -> tuple[str, int]:
    """[F2][F4] -> [S:key][S:key].  Returns (text, number_of_unknown_ids_dropped)."""
    dropped = 0

    def repl(m: re.Match) -> str:
        nonlocal dropped
        out = []
        for fid in re.findall(r"F\d+", m.group(1)):
            url = id_to_url.get(fid)
            if url:
                out.append(marker(url))
            else:
                dropped += 1
        return "".join(out)

    return _dedupe_adjacent(_FACT_GROUP.sub(repl, text)), dropped


def strip_fact_markers(text: str) -> str:
    return re.sub(r"\s*" + _FACT_GROUP.pattern, "", text)


def build_evidence(facts: Iterable, max_chars: int = 6000, per_source: int = 4) -> str:
    """Compact, keyed evidence listing for the Critic / reviser."""
    by_url: dict[str, list] = {}
    for f in facts:
        by_url.setdefault(f.source_url, []).append(f)
    lines: list[str] = []
    used = 0
    for url, fs in by_url.items():
        body = " | ".join(clean_untrusted(f.content).strip() for f in fs[:per_source])
        line = f"{marker(url)} {clean_untrusted(fs[0].source_title).strip()[:80]} — {body}"
        if used + len(line) > max_chars:
            lines.append("(evidence list truncated)")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if lines else "(no evidence available)"


def number_citations(sections: Sequence[str], citations: Iterable) -> tuple[list[str], list[dict], int]:
    """
    Replace [S:key] with [n] (first-appearance order) and return
    (numbered_sections, refs=[{n,url,title}], unknown_markers_removed).
    """
    by_key: dict[str, Any] = {}
    for c in citations:
        by_key.setdefault(source_key(c.url), c)
    order: dict[str, int] = {}
    unknown = 0

    def repl(m: re.Match) -> str:
        nonlocal unknown
        key = m.group(1)
        if key not in by_key:
            unknown += 1
            return ""
        if key not in order:
            order[key] = len(order) + 1
        return f"[{order[key]}]"

    numbered = [re.sub(r"\s+([.,;:])", r"\1", _SRC.sub(repl, s)) for s in sections]
    refs = [
        {"n": n, "url": by_key[k].url, "title": by_key[k].title}
        for k, n in sorted(order.items(), key=lambda kv: kv[1])
    ]
    return numbered, refs, unknown


def verify_numbered(report: str, valid: set[int]) -> tuple[str, int]:
    """Drop any [n] that does not correspond to a real reference."""
    removed = 0

    def repl(m: re.Match) -> str:
        nonlocal removed
        if int(m.group(1)) in valid:
            return m.group(0)
        removed += 1
        return ""

    return _NUM.sub(repl, report), removed


def cited_numbers(text: str) -> set[int]:
    return {int(n) for n in _NUM.findall(text)}


def render_references(refs: Sequence[dict], cited: set[int]) -> str:
    used = [r for r in refs if r["n"] in cited]
    if not used:
        return "## References\n\n_No claim in this report carries a source citation._"
    return "## References\n\n" + "\n".join(f"[{r['n']}] {r['title']} — {r['url']}" for r in used)


def analyze(sections: Sequence[str], facts: Iterable) -> dict:
    """
    Deterministic grounding report over [S:key]-marked text.

      uncited          claim-like sentences with no marker
      weakly_supported cited sentences sharing < 2 content tokens with the cited facts
      problems         up to 8 human-readable examples (fed to the Critic)
    """
    facts_by_key: dict[str, list[str]] = {}
    for f in facts:
        facts_by_key.setdefault(source_key(f.source_url), []).append(f.content)

    total = cited = uncited = weak = unknown = 0
    problems: list[str] = []
    for sec in sections:
        for sent in _SENT_SPLIT.split(sec.strip()):
            s = sent.strip()
            if len(s) < _MIN_SENTENCE_CHARS or s.startswith("#"):
                continue
            total += 1
            keys = _SRC.findall(s)
            if not keys:
                uncited += 1
                if len(problems) < 8:
                    problems.append(f"UNCITED: {s[:160]}")
                continue
            cited += 1
            known = [k for k in keys if k in facts_by_key]
            unknown += len(keys) - len(known)
            support = {t for k in known for c in facts_by_key[k] for t in tokenize(c)}
            shared = set(tokenize(_SRC.sub("", s))) & support
            if len(shared) < _MIN_SHARED_TOKENS:
                weak += 1
                if len(problems) < 8:
                    problems.append(f"WEAK SUPPORT (cited fact shares {len(shared)} key terms): {s[:160]}")
    return {
        "claim_sentences": total,
        "cited": cited,
        "uncited": uncited,
        "weakly_supported": weak,
        "unknown_markers": unknown,
        "cited_ratio": round(cited / total, 3) if total else 0.0,
        "problems": problems,
    }
