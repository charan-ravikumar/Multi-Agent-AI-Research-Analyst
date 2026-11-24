"""
core/gaps.py — turn the pipeline's raw "gaps" into something useful.

Two different things used to be mixed into one flat list:

  coverage       facts the SYSTEM detected: a search returned nothing, a branch failed, a synthesis
                 failed, no evidence could be extracted.  These affect how far to trust the report.
  open_question  the model's speculation about what the gathered sources did not address.  Useful as
                 *suggested follow-up research*, but unverified, repetitive across sub-questions
                 and, left uncurated, noise (a real run produced 20 for 5 sub-questions).

``curate_gaps`` keeps every coverage item, and reduces open questions to a short, deduplicated list
that covers each sub-question fairly (round-robin).
"""
from __future__ import annotations

from typing import Iterable

from research_analyst.retrieval.passages import tokenize

COVERAGE = "coverage"
OPEN_QUESTION = "open_question"


def make_item(sub_question: str, kind: str, text: str) -> dict:
    return {"sub_question": sub_question, "kind": kind, "text": text.strip()}


def _similar(a: set[str], b: set[str], threshold: float) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


def curate_gaps(items: Iterable[dict], max_open: int = 6, similarity: float = 0.5) -> dict:
    """-> {"coverage": [item...], "open_questions": [item...], "dropped": int}"""
    items = [i for i in items if i.get("text")]
    coverage: list[dict] = []
    seen_cov: set[str] = set()
    for i in items:
        if i["kind"] == COVERAGE and i["text"] not in seen_cov:
            seen_cov.add(i["text"])
            coverage.append(i)

    groups: dict[str, list[dict]] = {}
    for i in items:
        if i["kind"] == OPEN_QUESTION:
            groups.setdefault(i["sub_question"], []).append(i)

    chosen: list[dict] = []
    token_sets: list[set[str]] = []
    n_open = sum(len(g) for g in groups.values())
    depth = 0
    while len(chosen) < max_open and any(len(g) > depth for g in groups.values()):
        for g in groups.values():                       # round-robin: depth-th item of every sub-question
            if len(g) <= depth or len(chosen) >= max_open:
                continue
            toks = set(tokenize(g[depth]["text"]))
            if any(_similar(toks, t, similarity) for t in token_sets):
                continue
            chosen.append(g[depth])
            token_sets.append(toks)
        depth += 1
    return {"coverage": coverage, "open_questions": chosen, "dropped": n_open - len(chosen)}


def gap_texts(curated: dict) -> list[str]:
    """Flat strings for the Critic/Writer prompts."""
    return [c["text"] for c in curated["coverage"]] + [f"Open question: {o['text']}" for o in curated["open_questions"]]
