import asyncio

import httpx
import pytest

from config import settings
from models.search_result import SearchResult
from tests.helpers import FakeLLM, initial_state
from tools import arxiv as ax
from tools.query import parse_llm_queries, rules_query
from tools.sources import domain_of, rank_and_diversify, rrf_fuse, score_source


def R(url, title="t", snippet="s", sub="q"):
    return SearchResult(sub_question=sub, url=url, title=title, snippet=snippet)


# ── source scoring ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,lo,hi", [
    ("https://www.nih.gov/news/x", 0.9, 1.0),
    ("https://pubmed.ncbi.nlm.nih.gov/123/", 0.9, 1.0),
    ("https://cs.stanford.edu/paper", 0.9, 1.0),
    ("https://www.reuters.com/a", 0.7, 0.8),
    ("https://random-blog.example/post", 0.45, 0.55),
    ("https://www.pinterest.com/pin/1", 0.1, 0.25),
    ("https://www.academia.edu/123/paper", 0.45, 0.55),          # .edu TLD but a user-upload platform
    ("https://www.annualreviews.org/doi/x", 0.9, 1.0),
])
def test_domain_tiers(url, lo, hi):
    assert lo <= score_source(url, now_year=2026) <= hi


def test_recency_nudge():
    base = score_source("https://example.org/a", now_year=2026)
    assert score_source("https://example.org/a", "report 2025", now_year=2026) == pytest.approx(base + 0.05)
    assert score_source("https://example.org/a", "report 2012", now_year=2026) == pytest.approx(base - 0.10)


def test_domain_of_handles_two_part_suffixes():
    assert domain_of("https://news.bbc.co.uk/x") == "bbc.co.uk" and domain_of("https://www.a.b.example.com/p") == "example.com"


def test_rrf_rewards_agreement_between_queries():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]])
    assert fused["a"] > fused["b"] and fused["c"] > fused["b"]


def test_quality_beats_rank_and_domains_are_capped():
    res = [R("https://blog.example/1"), R("https://blog.example/2"), R("https://blog.example/3"),
           R("https://www.nih.gov/a"), R("https://www.pinterest.com/p")]
    fused = rrf_fuse([[r.url for r in res]])               # input order = original search order
    out = rank_and_diversify(res, fused, limit=4, quality_weight=0.7, max_per_domain=2)
    assert out[0].url == "https://www.nih.gov/a"            # high-quality source promoted from rank 4
    assert sum(1 for r in out if "blog.example" in r.url) == 2
    assert all(r.quality > 0 for r in out)


def test_diversity_overflow_fills_when_nothing_else_exists():
    res = [R(f"https://only.example/{i}") for i in range(4)]
    out = rank_and_diversify(res, rrf_fuse([[r.url for r in res]]), limit=4, max_per_domain=1)
    assert len(out) == 4


# ── query building ────────────────────────────────────────────────────────────

def test_rules_query_drops_question_words_and_keeps_topic_terms():
    q = rules_query("What are the regulatory challenges for solid-state batteries?", "electric vehicles")
    assert "regulatory" in q and "batteries" in q and "electric" in q and "what" not in q.split()


def test_llm_queries_are_validated():
    ok = parse_llm_queries('{"queries": ["solid state battery commercialization timeline review"]}', "How close are batteries?")
    assert ok == ["solid state battery commercialization timeline review"]
    assert parse_llm_queries('{"queries": ["tiny"]}', "q") == []
    assert parse_llm_queries("not json", "q") == []
    assert parse_llm_queries('{"queries": ["How close are batteries?"]}', "How close are batteries?") == []   # identical
    assert parse_llm_queries('{"queries": [123, null, "x"]}', "q") == []


# ── arXiv ─────────────────────────────────────────────────────────────────────

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Quantum error correction  with
 surface codes</title><summary>We study quantum error correction thresholds for surface codes and scaling limits.</summary></entry>
<entry><id>http://arxiv.org/abs/2401.00002v1</id><title>Cooking pasta</title><summary>Unrelated recipe paper.</summary></entry>
<entry><id>http://evil.example/abs/1</id><title>Bad host</title><summary>should be skipped</summary></entry>
</feed>"""


def test_arxiv_parse_and_relevance_gate():
    res = ax.parse_atom(ATOM, "What limits quantum error correction scaling?")
    assert [r.url for r in res] == ["https://arxiv.org/abs/2401.00001v1", "https://arxiv.org/abs/2401.00002v1"]
    assert res[0].source_type == "arxiv" and "surface codes" in res[0].full_text and "\n" not in res[0].title
    keep = [r for r in res if ax.relevant("What limits quantum error correction scaling?", r, 3)]
    assert [r.title for r in keep] == ["Quantum error correction with surface codes"]


def test_arxiv_search_request_shape_and_failure_modes(monkeypatch):
    monkeypatch.setattr(settings, "arxiv_min_interval_seconds", 0.0)
    seen = {}

    def handler(req):
        seen["q"] = req.url.params["search_query"]
        return httpx.Response(200, text=ATOM)

    c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = asyncio.run(ax.search_arxiv("quantum error correction scaling", sub_question="q", client=c))
    assert len(out) == 2 and "all:quantum" in seen["q"] and " AND " in seen["q"]
    bad = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert asyncio.run(ax.search_arxiv("quantum error correction", client=bad)) == []
    boom = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))))
    assert asyncio.run(ax.search_arxiv("quantum error correction", client=boom)) == []
    assert asyncio.run(ax.search_arxiv("the and of")) == []                  # no usable terms -> no request


def test_arxiv_requests_are_spaced(monkeypatch):
    monkeypatch.setattr(settings, "arxiv_min_interval_seconds", 0.2)
    ax._last_call = 0.0
    c = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=ATOM)))

    async def two():
        import time
        await ax.search_arxiv("quantum error correction", client=c)
        t = time.monotonic()
        await ax.search_arxiv("quantum error correction", client=c)
        return time.monotonic() - t

    assert asyncio.run(two()) >= 0.15


# ── Searcher integration ──────────────────────────────────────────────────────

def _state():
    st = initial_state()
    st["query"] = "solid-state batteries for electric vehicles"
    st["current_sub_question"] = "How close are solid-state batteries to commercial EVs?"
    return st


def _run_searcher(monkeypatch, search, rewrite="llm", arxiv_results=None, fake=None):
    from agents.searcher import SearcherAgent
    monkeypatch.setattr(settings, "query_rewrite", rewrite)
    monkeypatch.setattr(settings, "arxiv_enabled", bool(arxiv_results))
    monkeypatch.setattr(settings, "search_max_results", 5)
    monkeypatch.setattr("llm.generate", fake or FakeLLM())
    monkeypatch.setattr("agents.searcher.search_web", search)

    async def fake_arxiv(q, **kw):
        return arxiv_results or []

    monkeypatch.setattr("agents.searcher.search_arxiv", fake_arxiv)
    return asyncio.run(SearcherAgent()(_state()))


def test_searcher_runs_both_queries_fuses_ranks_and_caps_domains(monkeypatch):
    queries = []

    async def search(q, *, sub_question="", max_results=5):
        queries.append(q)
        if "rewritten" in q:
            return [R("https://www.nih.gov/a", sub=sub_question)]
        return [R(f"https://blog.example/{i}", sub=sub_question) for i in range(1, 5)]

    out = _run_searcher(monkeypatch, search)
    res = out["search_results"]
    assert len(queries) == 2 and queries[0].startswith("How close") and "rewritten" in queries[1]
    assert res[0].url == "https://www.nih.gov/a"                              # quality prior promotes it
    # diversity is a preference, not a hard cap: capped domains are ordered LAST and only fill spare slots
    assert sum(1 for r in res[:3] if "blog.example" in r.url) <= 2
    assert len({r.url for r in res}) == len(res)


def test_rewrite_failure_falls_back_to_rules(monkeypatch):
    queries = []

    async def search(q, *, sub_question="", max_results=5):
        queries.append(q)
        return [R(f"https://a.example/{len(queries)}", sub=sub_question)]

    fake = FakeLLM(overrides={"rewriter": lambda p, n: "garbage"})
    _run_searcher(monkeypatch, search, fake=fake)
    assert len(queries) == 2 and "batteries" in queries[1] and not queries[1].startswith("How")


def test_rewrite_off_makes_a_single_query(monkeypatch):
    queries = []

    async def search(q, *, sub_question="", max_results=5):
        queries.append(q)
        return [R("https://a.example/1", sub=sub_question)]

    fake = FakeLLM()
    _run_searcher(monkeypatch, search, rewrite="off", fake=fake)
    assert len(queries) == 1 and fake.total == 0


def test_relevant_arxiv_result_joins_the_pool_and_is_not_page_fetched(monkeypatch):
    async def search(q, *, sub_question="", max_results=5):
        return [R("https://blog.example/1", sub=sub_question)]

    paper = SearchResult(sub_question="x", url="https://arxiv.org/abs/2401.00001v1", title="Solid-state battery commercial EV review",
                         snippet="abstract", full_text="abstract text", source_type="arxiv")
    out = _run_searcher(monkeypatch, search, rewrite="off", arxiv_results=[paper])
    kinds = {r.source_type for r in out["search_results"]}
    assert kinds == {"web", "arxiv"}
    assert next(r for r in out["search_results"] if r.source_type == "arxiv").full_text == "abstract text"


def test_agreement_between_queries_can_outrank_a_better_domain(monkeypatch):
    """Documented trade-off: a mediocre page returned by BOTH queries beats a good page returned once."""
    async def search(q, *, sub_question="", max_results=5):
        if "rewritten" in q:
            return [R("https://www.nih.gov/a", sub=sub_question), R("https://blog.example/1", sub=sub_question)]
        return [R("https://blog.example/1", sub=sub_question)]

    out = _run_searcher(monkeypatch, search)
    assert out["search_results"][0].url == "https://blog.example/1"


def test_arxiv_429_triggers_cooldown_and_stops_calling(monkeypatch):
    monkeypatch.setattr(settings, "arxiv_min_interval_seconds", 0.0)
    ax._cooldown_until = 0.0
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(429, text="Rate exceeded.")

    c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert asyncio.run(ax.search_arxiv("quantum error correction", client=c)) == []
    assert asyncio.run(ax.search_arxiv("quantum error correction", client=c)) == []
    assert calls["n"] == 1                               # second call skipped during cooldown
    ax._cooldown_until = 0.0                             # do not leak into other tests
