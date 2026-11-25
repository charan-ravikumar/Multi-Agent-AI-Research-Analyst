"""
Shared pytest fixtures.  Everything here is offline: no network, no real keys,
no real Redis.  Environment is pinned BEFORE any project import so config.Settings
never reads real credentials and the trace log goes to a temp dir.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="research-analyst-tests-")
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["GEMINI_API_KEY_2"] = ""                # never pick up a real second key from .env
os.environ["GEMINI_LIMIT_HEADROOM"] = "1.0"       # tests assert exact limits; headroom has its own test
os.environ["GEMINI_DAILY_RESERVE"] = "0"
os.environ["LOG_FILE"] = os.path.join(_TMP, "agent_traces.jsonl")
os.environ["LANGFUSE_PUBLIC_KEY"] = ""      # never send test traces, even if .env has real keys
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_BASE_URL"] = ""
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
os.environ["TOKEN_BUDGET_PER_RUN"] = "0"
os.environ["LLM_PRICE_PER_1M_TOKENS"] = ""
os.environ["QUERY_REWRITE"] = "off"             # evidence-quality tests opt in explicitly
os.environ["ARXIV_ENABLED"] = "false"
os.environ["DEBATE_MAX_ROUNDS"] = "3"          # review loop is opt-in in production defaults; tests exercise it
os.environ["GEMINI_RATE_LIMITS"] = "{}"           # unit tests of the client opt in explicitly
os.environ["RUNS_FILE"] = os.path.join(_TMP, "runs.jsonl")
os.environ["LLM_QUOTA_FILE"] = os.path.join(_TMP, "llm_quota.json")
os.environ["GEMINI_STRONG_AGENTS"] = "writer"
os.environ["JUDGE_MODEL"] = ""
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"   # unreachable on purpose

import fakeredis  # noqa: E402
import pytest  # noqa: E402


from tests.helpers import FakeLLM, initial_state  # noqa: E402,F401

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Route every redis_store call to an in-process fakeredis."""
    import research_analyst.memory.redis_store as rs
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(rs, "_client", client)
    return client


@pytest.fixture(autouse=True)
def no_page_fetching(monkeypatch):
    """Offline tests never fetch pages; tests of fetching call tools.fetch directly."""
    async def _none(urls, **kw):
        return {}
    async def _no_arxiv(*a, **kw):
        return []

    monkeypatch.setattr("research_analyst.agents.searcher.fetch_many", _none)
    async def _no_pdf(*a, **kw):
        return None

    monkeypatch.setattr("research_analyst.agents.searcher.search_arxiv", _no_arxiv)
    monkeypatch.setattr("research_analyst.agents.searcher.fetch_pdf_text", _no_pdf)


@pytest.fixture(autouse=True)
def reset_llm_state():
    from research_analyst.observability import tracing
    from research_analyst.observability.usage import tracker
    from research_analyst.llm import client
    tracing.reset_for_tests()
    from research_analyst.observability import prompt_store
    prompt_store.reset_for_tests()
    tracker.reset()
    from research_analyst.retrieval import arxiv as _ax
    _ax._cache.clear()
    _ax._cooldown_until = 0.0
    _ax._last_call = 0.0
    client._breaker.reset()
    Path(os.environ["LLM_QUOTA_FILE"]).unlink(missing_ok=True)      # daily-request counter is per test
    Path(os.environ["RUNS_FILE"]).unlink(missing_ok=True)           # run history is per test
    from research_analyst.observability import metrics
    metrics.reset()
    client._tpm.reset()


@pytest.fixture
def run_graph(monkeypatch):
    """run_graph(fake_llm, search_fn, auto_approve=True) -> (final_state, fake_llm)"""
    from research_analyst.pipeline.graph import app

    def _run(fake: FakeLLM, search, auto_approve: bool = True, review_rounds: int | None = None):
        monkeypatch.setattr("research_analyst.llm.generate", fake)
        monkeypatch.setattr("research_analyst.agents.searcher.search_web", search)
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        st = initial_state(auto_approve)
        if review_rounds is not None:
            st["review_rounds"] = review_rounds
        return asyncio.run(app.ainvoke(st, config=cfg))

    return _run
