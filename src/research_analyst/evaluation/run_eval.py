"""
Run the golden-set evaluation.   LIVE: uses real LLM/search quota unless fixtures + cache exist.

    uv run python -m eval.run_eval --queries 3 --configs no_debate,debate --max-new-tokens 120000

Safety rails for free tiers: --max-new-tokens counts only *uncached* tokens and aborts the whole
evaluation when exceeded; a provider "per day" quota error aborts immediately.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from research_analyst import llm  # noqa: E402
from research_analyst.config import settings  # noqa: E402
from research_analyst.evaluation import harness as H  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=3, help="first N golden queries")
    ap.add_argument("--configs", default="no_debate,debate")
    ap.add_argument("--max-new-tokens", type=int, default=120_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from research_analyst.llm.preflight import assert_models_available
    assert_models_available()

    queries = json.loads((H.ROOT / "golden_queries.json").read_text("utf-8"))[: a.queries]
    configs = a.configs.split(",")
    for c in configs:
        if c not in H.CONFIGS:
            raise SystemExit(f"unknown config {c}; choose from {list(H.CONFIGS)}")

    from research_analyst.retrieval.arxiv import search_arxiv
    from research_analyst.retrieval.fetch import fetch_many
    from research_analyst.retrieval.web_search import search_web
    store = H.FixtureStore(live_search=search_web, live_fetch=fetch_many, live_arxiv=search_arxiv)
    cached = H.CachedLLM(llm.generate)
    import research_analyst.agents.searcher as S
    llm.generate = cached
    S.search_web, S.fetch_many, S.search_arxiv = store.search, store.fetch, store.arxiv

    rows, errors = [], []
    for q in queries:
        for cfg in configs:
            if cached.miss_tokens > a.max_new_tokens:
                print(f"STOP: new-token cap {a.max_new_tokens} reached")
                break
            t0 = time.time()
            try:
                res = await H.run_one(q["query"], cfg)
                rows.append({"query_id": q["id"], "config": cfg, "metrics": res["metrics"]})
                m = res["metrics"]
                print(f"[{q['id']}/{cfg}] {time.time()-t0:.0f}s  cited={m['cited_ratio']} fidelity={m['fact_fidelity']} "
                      f"judge_faith={m['judge_faithfulness']} complete={m['judge_completeness']} "
                      f"| llm cache hits={cached.hits} misses={cached.misses} new_tokens={cached.miss_tokens}", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append({"query_id": q["id"], "config": cfg, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                print(f"[{q['id']}/{cfg}] FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                if "per day" in str(exc).lower() or "quota" in str(exc).lower():
                    print("STOP: daily quota error")
                    break

    base, other = configs[0], configs[-1]
    agg = H.aggregate(rows, base, other) if len(configs) > 1 else {}
    meta = {"groq_models": f"{settings.groq_fast_model} / {settings.groq_strong_model}", "judge_model": settings.judge_model,
            "new_tokens": cached.miss_tokens, "llm_cache_hits": cached.hits, "llm_cache_misses": cached.misses}
    H.RESULTS.mkdir(exist_ok=True)
    out = Path(a.out) if a.out else H.RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows, "errors": errors, "aggregate": agg}, indent=1, default=str), "utf-8")
    if agg:
        (H.ROOT / "RESULTS.md").write_text(H.to_markdown(agg, base, other, meta), "utf-8")
    print(f"\nwrote {out}  | new tokens spent: {cached.miss_tokens}  errors: {len(errors)}")
    return 1 if errors and not rows else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
