"""
Offline proxy check (no network, no LLM): do the ranking + diversification rules change what the
Reader would see, on the REAL search results recorded in eval/fixtures?

    uv run python -m eval.source_quality_check

Reports, per recorded sub-question search, the mean heuristic quality of the top-K results before
(original search order) vs after ``rank_and_diversify``, and the largest same-domain count in the top-K.
This measures the *source-selection prior only*; it says nothing about final answer quality.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import FIXTURES  # noqa: E402
from models.search_result import SearchResult  # noqa: E402
from tools.sources import domain_of, rank_and_diversify, rrf_fuse, score_source  # noqa: E402

K = 3


def main() -> None:
    rows = []
    for p in sorted(FIXTURES.glob("search_*.json")):
        res = [SearchResult(**r) for r in json.loads(p.read_text("utf-8"))]
        if len(res) < K + 1:
            continue
        before = res[:K]
        after = rank_and_diversify(res, rrf_fuse([[r.url for r in res]]), limit=len(res))[:K]
        q = lambda xs: statistics.mean(score_source(r.url, r.title, r.snippet) for r in xs)  # noqa: E731
        dom = lambda xs: max(Counter(domain_of(r.url) for r in xs).values())  # noqa: E731
        rows.append((q(before), q(after), dom(before), dom(after), [domain_of(r.url) for r in res[:K]], [domain_of(r.url) for r in after]))
    n = len(rows)
    print(f"{n} recorded searches (top-{K})")
    print(f"mean source quality  before {statistics.mean(r[0] for r in rows):.3f}  ->  after {statistics.mean(r[1] for r in rows):.3f}")
    print(f"improved/same/worse  {sum(r[1] > r[0] + 1e-9 for r in rows)}/{sum(abs(r[1]-r[0]) <= 1e-9 for r in rows)}/{sum(r[1] < r[0] - 1e-9 for r in rows)}")
    print(f"max same-domain in top-{K}: before {statistics.mean(r[2] for r in rows):.2f}  ->  after {statistics.mean(r[3] for r in rows):.2f}")
    print("domains before vs after (first 5 searches):")
    for r in rows[:5]:
        print("  ", r[4], "->", r[5])


if __name__ == "__main__":
    main()
