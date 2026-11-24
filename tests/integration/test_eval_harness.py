import asyncio
import json

from research_analyst.evaluation import harness as H
from tests.helpers import FakeLLM, make_search


def _judge_llm(claims_ok=True, provider="groq", model="qwen/qwen3.8-27b"):
    def gen(prompt, *, return_usage=False, **kw):
        n = prompt[1]["content"].count("CLAIM:")
        verdicts = [{"id": i, "verdict": "supported" if claims_ok else "unsupported"} for i in range(1, n + 1)]
        out = json.dumps({"claims": verdicts, "completeness": 4, "clarity": 5})
        return (out, {"provider": provider, "model": model, "total_tokens": 10}) if return_usage else out
    return gen


def _cited_fake():
    return FakeLLM(4, {
        "reader": lambda p, n: json.dumps({"facts": [{"source": 1, "content": "Machine learning shortens drug screening.", "confidence": 0.9}]}),
        "synth": lambda p, n: json.dumps({"synthesis": "Machine learning shortens early drug screening cycles. [F1]", "contradictions": [], "gaps": []}),
        "writer": lambda p, n: ("## Executive Summary\nShort.\n## Key Findings\n- Machine learning shortens early drug screening cycles [1]\n"
                                "## Detailed Analysis\nMachine learning shortens early drug screening cycles for many teams [1].\n"
                                "## Contradictions and Conflicting Evidence\nNone.\n## Knowledge Gaps and Limitations\n- limits"),
    })


def test_cached_llm_hits_on_identical_request_and_counts_new_tokens(tmp_path):
    inner = FakeLLM(4)
    c = H.CachedLLM(inner, tmp_path)
    msg = [{"role": "system", "content": "research writer"}, {"role": "user", "content": "x"}]
    a = c(msg, return_usage=True, json_mode=False)
    b = c(msg, return_usage=True, json_mode=False)
    assert a == b and c.hits == 1 and c.misses == 1 and inner.total == 1 and c.miss_tokens == 2
    c(msg, return_usage=True, json_mode=True)                       # different request -> miss
    assert c.misses == 2


def test_fixture_store_records_then_replays(tmp_path):
    calls = {"s": 0, "f": 0}
    live = make_search(2)

    async def live_search(q, **kw):
        calls["s"] += 1
        return await live(q, **kw)

    async def live_fetch(urls, **kw):
        calls["f"] += 1
        return dict.fromkeys(urls, "page text " * 20)

    st = H.FixtureStore(tmp_path, live_search, live_fetch)
    r1 = asyncio.run(st.search("q", sub_question="q"))
    r2 = asyncio.run(st.search("q", sub_question="q"))
    f1 = asyncio.run(st.fetch([r1[0].url]))
    f2 = asyncio.run(st.fetch([r1[0].url]))
    assert [x.url for x in r1] == [x.url for x in r2] and f1 == f2 and calls == {"s": 1, "f": 1}


def test_run_one_scores_a_report_and_restores_settings(monkeypatch, tmp_path):
    from research_analyst.config import settings
    fake = _cited_fake()
    cached = H.CachedLLM(fake, tmp_path)
    monkeypatch.setattr("research_analyst.llm.generate", cached)
    monkeypatch.setattr("research_analyst.agents.searcher.search_web", make_search(2, snippet="Machine learning shortens drug screening cycles a lot."))
    before = settings.debate_max_rounds
    res = asyncio.run(H.run_one("What is X?", "no_debate", generate=_judge_llm()))
    m = res["metrics"]
    assert settings.debate_max_rounds == before
    assert m["judge_faithfulness"] == 1.0 and m["judge_independent"] is True and m["judge_completeness"] == 4
    assert m["fact_fidelity"] == 1.0 and m["cited_ratio"] == 1.0 and m["debate_rounds"] == 0


def test_judge_flags_same_family_model_as_not_independent(monkeypatch):
    state = {"query": "q", "final_report": "## A\nMachine learning shortens early drug screening cycles a lot [1].\n## References\n[1] t - u",
             "references": [{"n": 1, "url": "https://a", "title": "t"}], "extracted_facts": []}
    out = H.judge_report(state, _judge_llm(model="openai/gpt-oss-120b"))
    assert out["judge_independent"] is False and out["unsupported_claims"] == 0


def test_second_config_reuses_cached_pipeline(monkeypatch, tmp_path):
    fake = _cited_fake()
    cached = H.CachedLLM(fake, tmp_path)
    monkeypatch.setattr("research_analyst.llm.generate", cached)
    monkeypatch.setattr("research_analyst.agents.searcher.search_web", make_search(2))
    asyncio.run(H.run_one("What is X?", "no_debate", generate=_judge_llm()))
    misses_after_first = cached.misses
    asyncio.run(H.run_one("What is X?", "debate", generate=_judge_llm()))
    # planner/reader/synth/writer prompts are identical when the critic approves -> only the critic call is new
    assert cached.misses - misses_after_first <= 2 and cached.hits >= 8


def test_aggregate_paired_deltas():
    rows = []
    for q, (a, b) in {"q1": (0.5, 0.8), "q2": (0.7, 0.7), "q3": (0.9, 0.6)}.items():
        rows.append({"query_id": q, "config": "no_debate", "metrics": {"judge_faithfulness": a, "uncited": 3, "judge_independent": True}})
        rows.append({"query_id": q, "config": "debate", "metrics": {"judge_faithfulness": b, "uncited": 1 if q != "q3" else 5, "judge_independent": True}})
    agg = H.aggregate(rows, "no_debate", "debate")
    p = agg["paired"]["judge_faithfulness"]
    assert (p["other_better"], p["other_worse"], p["ties"], p["n"]) == (1, 1, 1, 3) and p["mean_delta"] == 0.0
    u = agg["paired"]["uncited"]                                    # lower is better: 2 queries improved
    assert (u["other_better"], u["other_worse"]) == (2, 1)
    md = H.to_markdown(agg, "no_debate", "debate", {"groq_models": "g", "judge_model": "j"})
    assert "Paired difference" in md and "judge model family differs from the generator in 6/6" in md


def test_cached_llm_does_not_forward_none_args(tmp_path):
    seen = {}

    def inner(prompt, **kw):
        seen.update(kw)
        return "ok", {"total_tokens": 1}

    H.CachedLLM(inner, tmp_path)("hello", return_usage=True)
    assert "tier" not in seen and "prefer" not in seen and seen["return_usage"] is True


def test_truncated_judge_is_an_error_not_a_zero_score():
    import pytest

    def truncated(prompt, *, return_usage=False, **kw):
        return ('{"claims": [{"id": 1, "verdict": "supp', {"provider": "groq", "model": "qwen/qwen3.8-27b", "finish_reason": "length"})

    state = {"query": "q", "final_report": "## A\nMachine learning shortens early drug screening cycles a lot [1].\n## References\n[1] t - u",
             "references": [{"n": 1, "url": "https://a", "title": "t"}], "extracted_facts": []}
    with pytest.raises(ValueError, match="truncated"):
        H.judge_report(state, truncated)


def test_truncated_responses_are_not_cached(tmp_path):
    inner = lambda p, **kw: ("cut", {"finish_reason": "length", "total_tokens": 5})  # noqa: E731
    c = H.CachedLLM(inner, tmp_path)
    c("x", return_usage=True)
    c("x", return_usage=True)
    assert c.misses == 2 and c.hits == 0


def test_non_independent_judge_scores_are_excluded_from_aggregates():
    rows = [
        {"query_id": "q1", "config": "a", "metrics": {"judge_faithfulness": 0.9, "judge_independent": True}},
        {"query_id": "q1", "config": "b", "metrics": {"judge_faithfulness": 0.1, "judge_independent": False}},
    ]
    agg = H.aggregate(rows, "a", "b")
    assert agg["paired"]["judge_faithfulness"]["n"] == 0 and agg["configs"]["b"]["judge_faithfulness"] is None


def test_family_helper():
    assert H._family("openai/gpt-oss-120b") == "openai" and H._family("gemini-2.5-flash") == "google"
    assert H._independent("qwen/qwen3.8-27b") and not H._independent("openai/gpt-oss-20b")
