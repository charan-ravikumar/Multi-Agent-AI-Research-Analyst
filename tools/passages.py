"""
tools/passages.py — split page text into passages and rank them against the
sub-question with BM25.  No embeddings, no vector store: a page has a handful of
passages, so lexical BM25 over them is cheap, deterministic and testable.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset(
    "the and for are was were has have had its this that these those with from into over than then "
    "what how why when where which who does did been being will would could should may might can but not "
    "all any about also more most some such their there they them your you our out use used using".split()
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def split_passages(text: str, target: int = 500, max_len: int = 900) -> list[str]:
    """Paragraph-aware chunks of roughly ``target`` chars (never > ``max_len``)."""
    chunks: list[str] = []
    cur = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n|\n", text)):
        if not para:
            continue
        pieces = [para]
        if len(para) > max_len:                       # split oversized paragraphs on sentence boundaries
            pieces, buf = [], ""
            for s in _SENT_SPLIT.split(para):
                if buf and len(buf) + len(s) + 1 > max_len:
                    pieces.append(buf)
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                pieces.append(buf)
        for piece in pieces:
            if cur and len(cur) + len(piece) + 1 > target:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n{piece}".strip()
    if cur:
        chunks.append(cur)
    return [c[:max_len] for c in chunks if len(c) >= 40]


def bm25_rank(query: str, passages: list[str], k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """Return [(passage_index, score)] best-first.  Zero-score passages are dropped."""
    q = set(tokenize(query))
    docs = [tokenize(p) for p in passages]
    if not q or not docs:
        return []
    n = len(docs)
    avg = (sum(len(d) for d in docs) / n) or 1.0
    df = Counter(t for d in docs for t in set(d) if t in q)
    scores: list[tuple[int, float]] = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        if s > 0:
            scores.append((i, s))
    return sorted(scores, key=lambda x: -x[1])


def select_passages(query: str, text: str, k: int = 3, max_chars: int = 1200, target: int = 350) -> str:
    """Top-``k`` BM25 passages (kept in page order for readability), capped at ``max_chars``."""
    passages = split_passages(text, target=target)
    top = [i for i, _ in bm25_rank(query, passages)[:k]]
    out, used = [], 0
    for i in sorted(top):
        p = passages[i]
        if used + len(p) > max_chars:
            p = p[: max(0, max_chars - used)]
        if not p:
            break
        out.append(p)
        used += len(p)
    return "\n…\n".join(out)
