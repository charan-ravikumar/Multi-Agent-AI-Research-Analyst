"""
tests/test_memory_across_runs.py  —  Verification 5

Tests cross-run Redis memory: storing completed reports and surfacing topically
related work for subsequent queries.

REDIS NOTE
----------
Real Redis is NOT available on this machine (org policy blocks installation).
This test uses fakeredis 2.36.2 — an in-process drop-in that implements the
same Redis command set.  The production code paths (save_report_record,
find_related_reports, get_client) run unchanged; only the client singleton is
substituted via module-level injection before any Redis call is made.

  fakeredis docs: https://github.com/cunla/fakeredis-py

  Production swap: in deployment config, remove the `rs._client = ...` line
  and let `get_client()` connect to a real Redis server via settings.redis_url.

MATCHING LIMITATIONS (documented — not a bug)
----------------------------------------------
The lookup uses keyword (token) overlap, NOT semantic similarity:

  False positives:
    Queries sharing generic domain vocabulary ("drug", "cancer", "model")
    may match even when the actual research questions differ.

  False negatives:
    Semantically related queries with different surface forms will NOT match.
    Example: "ML in pharma" vs "AI in drug discovery" — no 3-char shared
    tokens → zero overlap → missed connection.  Short acronyms (AI, ML) are
    excluded by the 3-character minimum.

  Deliberate exclusion:
    Semantic search (cosine similarity over embeddings) is out of scope for
    Stage 5 (no vector store). A future upgrade could add Redis Stack
    RediSearch vector field without changing find_related_reports' signature.

WHAT WE VERIFY (10 assertions)
-------------------------------
Run 1  (query: "What is the impact of AI on drug discovery timelines?"):
  R1-1  final_report is produced.
  R1-2  Redis reports:index has exactly 1 entry after the run.
  R1-3  related_past_reports is [] (no prior history at run start).

Run 2  (query: "Applications of machine learning in pharmaceutical drug discovery"):
  R2-1  final_report is produced.
  R2-2  At least 1 related report surfaced (topic overlap with Run 1).
  R2-3  The surfaced report is Run 1's record.
  R2-4  Overlap tokens include "drug" and "discovery" (the shared keywords).
  R2-5  Redis reports:index has exactly 2 entries after the run.

Run 3  (query: "Effects of climate change on Arctic ecosystem biodiversity"):
  R3-1  final_report is produced.
  R3-2  related_past_reports is [] — unrelated topic, no keyword overlap.
        Note: "effects" and "impact" are in the stopword list so they do not
        cause false positives between Runs 1/3.  "AI"/"ML" are also excluded
        (2-char tokens filtered by the tokenizer).

Run with:
    .venv/Scripts/python.exe tests/test_memory_across_runs.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── fakeredis injection (must happen before any redis_store call) ─────────────
import fakeredis
import memory.redis_store as rs

# Pre-set the module-level singleton.  get_client() checks `if _client is None`
# and returns the existing client immediately — it will never attempt a real
# Redis connection for the lifetime of this test process.
_fake_redis = fakeredis.FakeRedis(decode_responses=True)
rs._client = _fake_redis
# ─────────────────────────────────────────────────────────────────────────────

from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.reader import ReaderAgent
from agents.searcher import SearcherAgent
from agents.synthesizer import SynthesizerAgent
from agents.writer import WriterAgent
from models import ResearchState
from models.fact import Fact
from models.research_plan import ResearchPlan
from models.search_result import SearchResult
from orchestrator.graph import app

# ── Mock agent implementations ────────────────────────────────────────────────

async def _mock_planner(self, state: ResearchState) -> dict:
    # Sub-questions deliberately include topically relevant words from the query
    # so that sub_questions also contribute to the overlap search in Run 2.
    plan = ResearchPlan(
        query=state["query"],
        sub_questions=[
            f"How does this domain apply to: {state['query'][:40]}?",
            f"What are the limitations of: {state['query'][:40]}?",
            f"What evidence exists for: {state['query'][:40]}?",
        ],
    )
    return {**state, "research_plan": plan, "sub_questions": plan.sub_questions}


async def _mock_searcher(self, state: ResearchState) -> dict:
    sq = state.get("current_sub_question", "")
    result = SearchResult(
        sub_question=sq, url="https://example.com/1",
        title="Mock result", snippet=f"Mock snippet for {sq[:40]}",
    )
    return {"search_results": [result], "failed_sub_questions": []}


async def _mock_reader(self, state: ResearchState) -> dict:
    sq = state.get("current_sub_question", "")
    fact = Fact(
        sub_question=sq, content=f"Mock fact for {sq}",
        source_url="https://example.com/1", source_title="Mock Source",
        confidence=0.9,
    )
    return {"extracted_facts": [fact]}


async def _mock_synthesizer(self, state: ResearchState) -> dict:
    if state.get("critic_objections"):
        return {"revised_draft": ["[MOCK REVISION]"]}
    sq = state.get("current_sub_question", "?")
    return {
        "draft": [f"[MOCK SYNTHESIS: {sq}]"],
        "citations": [], "contradictions": [], "unresolved_gaps": [],
    }


async def _mock_critic(self, state: ResearchState) -> dict:
    return {"critic_objections": []}   # organic resolution — no debate


async def _mock_writer(self, state: ResearchState) -> dict:
    sections = state.get("revised_draft") or state.get("draft", [])
    return {"final_report": f"[MOCK REPORT] sections={len(sections)}  query={state.get('query','?')[:50]}"}


# ── helper: run one graph pass with all agents mocked ────────────────────────

async def run_graph(query: str) -> dict:
    initial = ResearchState(
        session_id=str(uuid.uuid4()),
        query=query,
        research_plan=None,
        sub_questions=[],
        search_results=[],
        failed_sub_questions=[],
        extracted_facts=[],
        citations=[],
        contradictions=[],
        unresolved_gaps=[],
        draft=[],
        critique=None,
        final_report="",
        iteration=0,
        routing_history=[],
        auto_approve_plan=True,   # skip human checkpoint; lookup still runs
    )
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    with (
        patch.object(PlannerAgent,     "run", new=_mock_planner),
        patch.object(SearcherAgent,    "run", new=_mock_searcher),
        patch.object(ReaderAgent,      "run", new=_mock_reader),
        patch.object(SynthesizerAgent, "run", new=_mock_synthesizer),
        patch.object(CriticAgent,      "run", new=_mock_critic),
        patch.object(WriterAgent,      "run", new=_mock_writer),
    ):
        return await app.ainvoke(initial, config=cfg)


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "=" * 72)
    print("VERIFICATION 5 — Cross-Run Redis Memory (fakeredis in-process)")
    print("=" * 72)
    print()
    print("Redis backend : fakeredis.FakeRedis (injected via rs._client)")
    print("Real Redis    : NOT available (org policy blocks installation)")
    print("Production    : remove rs._client injection; get_client() will")
    print("                connect to settings.redis_url automatically.")
    print()

    failures: list[str] = []
    assertions_total = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal assertions_total
        assertions_total += 1
        if cond:
            print(f"  PASS \u2713  [{label}]" + (f"  \u2014  {detail}" if detail else ""))
        else:
            msg = f"[{label}]" + (f" {detail}" if detail else "")
            print(f"  FAIL \u2717  {msg}")
            failures.append(msg)

    # ── RUN 1 ─────────────────────────────────────────────────────────────────
    QUERY_1 = "What is the impact of AI on drug discovery timelines?"
    print(f"\u2500\u2500 RUN 1 {'─' * 63}")
    print(f"Query : {QUERY_1}")

    result1 = await run_graph(QUERY_1)

    idx_size_1 = _fake_redis.zcard("reports:index")
    related_1: list = result1.get("related_past_reports", [])

    print(f"  final_report         : {bool(result1.get('final_report'))}")
    print(f"  related_past_reports : {related_1}")
    print(f"  Redis index size     : {idx_size_1}")

    check("R1-1", bool(result1.get("final_report")), "final_report present")
    check("R1-2", idx_size_1 == 1, f"reports:index has 1 entry (got {idx_size_1})")
    check("R1-3", related_1 == [],
          f"related_past_reports is [] — no prior history (got {len(related_1)})")

    # ── RUN 2 ─────────────────────────────────────────────────────────────────
    QUERY_2 = "Applications of machine learning in pharmaceutical drug discovery"
    print(f"\n\u2500\u2500 RUN 2 {'─' * 63}")
    print(f"Query : {QUERY_2}")
    print(f"  (Expected match with Run 1 via shared tokens: 'drug', 'discovery')")

    result2 = await run_graph(QUERY_2)

    idx_size_2 = _fake_redis.zcard("reports:index")
    related_2: list = result2.get("related_past_reports", [])

    print(f"  final_report         : {bool(result2.get('final_report'))}")
    print(f"  Redis index size     : {idx_size_2}")
    print(f"  related_past_reports ({len(related_2)} match(es)):")
    for r in related_2:
        print(f"    query    : {r.get('query', '?')[:70]}")
        print(f"    overlap  : {r.get('_overlap_count', 0)} token(s) — {r.get('_overlap_tokens', [])}")

    check("R2-1", bool(result2.get("final_report")), "final_report present")
    check("R2-2", len(related_2) >= 1,
          f"at least 1 related report surfaced (got {len(related_2)})")

    # Inspect the best match (may not exist if R2-2 failed, handled gracefully)
    best = related_2[0] if related_2 else {}
    check("R2-3", best.get("query") == QUERY_1,
          f"surfaced record is Run 1's (got: {best.get('query', 'N/A')[:50]})")
    overlap_set = set(best.get("_overlap_tokens", []))
    check("R2-4", {"drug", "discovery"}.issubset(overlap_set),
          f"overlap includes 'drug' + 'discovery' (got: {sorted(overlap_set)})")
    check("R2-5", idx_size_2 == 2, f"reports:index has 2 entries (got {idx_size_2})")

    # ── RUN 3 ─────────────────────────────────────────────────────────────────
    QUERY_3 = "Effects of climate change on Arctic ecosystem biodiversity"
    print(f"\n\u2500\u2500 RUN 3 {'─' * 63}")
    print(f"Query : {QUERY_3}")
    print(f"  (Expected: no match — 'effects'/'impact' are stopwords,")
    print(f"   no other token overlap with Runs 1 or 2)")

    result3 = await run_graph(QUERY_3)

    related_3: list = result3.get("related_past_reports", [])
    print(f"  final_report         : {bool(result3.get('final_report'))}")
    print(f"  related_past_reports : {related_3}")

    check("R3-1", bool(result3.get("final_report")), "final_report present")
    check("R3-2", related_3 == [],
          f"no matches for unrelated topic (got {len(related_3)} unexpected match(es))")

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = assertions_total - len(failures)
    print("\n" + "─" * 72)
    if not failures:
        print(f"ALL {assertions_total} ASSERTIONS PASSED  ({passed}/{assertions_total})")
        print()
        print("Memory layer summary:")
        print(f"  save_report_record : stores compact record in 'reports:index'")
        print(f"                       (sorted set, score=Unix timestamp)")
        print(f"  find_related_reports: keyword overlap, min 2 shared tokens")
        print(f"                       _tokenize filters 3-char min + stopwords")
        print(f"  Known limitation   : 2-char acronyms (AI, ML) are excluded")
        print(f"  Known limitation   : false negatives on synonym variation")
        print(f"  Known limitation   : false positives on shared domain terms")
        print(f"  Production ready   : swap fakeredis for real Redis via settings.redis_url")
    else:
        print(f"FAILED: {len(failures)}/{assertions_total} assertion(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
