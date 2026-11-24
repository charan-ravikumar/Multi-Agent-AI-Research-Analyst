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
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
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
os.environ["GROQ_FALLBACK_MODEL"] = ""        # unit tests of the client opt in explicitly
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"   # unreachable on purpose

import fakeredis  # noqa: E402
import pytest  # noqa: E402


from tests.helpers import FakeLLM, initial_state  # noqa: E402,F401

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Route every redis_store call to an in-process fakeredis."""
    import memory.redis_store as rs
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

    monkeypatch.setattr("agents.searcher.fetch_many", _none)
    monkeypatch.setattr("agents.searcher.search_arxiv", _no_arxiv)


@pytest.fixture(autouse=True)
def reset_llm_state():
    from core import tracing
    from core.usage import tracker
    from llm import client
    tracing.reset_for_tests()
    tracker.reset()
    from tools import arxiv as _ax
    _ax._cache.clear()
    _ax._cooldown_until = 0.0
    _ax._last_call = 0.0
    client._breaker.reset()
    client._tpm.reset()


@pytest.fixture
def run_graph(monkeypatch):
    """run_graph(fake_llm, search_fn, auto_approve=True) -> (final_state, fake_llm)"""
    from orchestrator.graph import app

    def _run(fake: FakeLLM, search, auto_approve: bool = True, review_rounds: int | None = None):
        monkeypatch.setattr("llm.generate", fake)
        monkeypatch.setattr("agents.searcher.search_web", search)
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        st = initial_state(auto_approve)
        if review_rounds is not None:
            st["review_rounds"] = review_rounds
        return asyncio.run(app.ainvoke(st, config=cfg))

    return _run
