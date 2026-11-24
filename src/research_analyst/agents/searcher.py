from __future__ import annotations

import asyncio
from typing import Any

from research_analyst.agents.base import BaseAgent, StateLike
from research_analyst.agents.react_search import ReactSearch, url_key as _url_key
from research_analyst.config import settings
from research_analyst.observability.usage import BudgetExceeded
from research_analyst.llm import ModelTier
from research_analyst.models.search_result import SearchResult
from research_analyst.retrieval.arxiv import fetch_pdf_text
from research_analyst.retrieval.arxiv import relevant as arxiv_relevant
from research_analyst.retrieval.arxiv import search_arxiv
from research_analyst.retrieval.fetch import fetch_many
from research_analyst.retrieval.passages import select_passages
from research_analyst.retrieval.query import REWRITE_SYSTEM, parse_llm_queries, rewrite_user_prompt, rules_query
from research_analyst.retrieval.sources import rank_and_diversify, rrf_fuse
from research_analyst.retrieval.web_search import search_web


class SearcherError(RuntimeError):
    """Raised when the Searcher cannot determine which sub-question to process."""


class SearcherAgent(BaseAgent):
    """
    Gathers evidence candidates for ONE sub-question.

    1. Queries: the verbatim sub-question plus (``settings.query_rewrite``) one rewritten
       keyword query ("llm": a ~300-token FAST-model call; "rules": free keyword extraction).
    2. Search: DuckDuckGo for every query (+ arXiv abstracts when clearly relevant).
    3. Rank: reciprocal-rank fusion across queries, blended with a heuristic source-quality
       prior, then a per-domain cap so one site cannot supply all the evidence
       (tools/sources.py).  Unscored/low-quality sources sink, they are not removed.
    4. Fetch: the top ``fetch_top_n`` pages (SSRF-guarded, bounded) are reduced to their BM25
       best passages for the sub-question and stored in ``SearchResult.full_text``; fetch
       failures silently keep the snippet.

    Reads  : state["current_sub_question"], state["query"]
    Writes : state["search_results"], state["failed_sub_questions"] (when zero results)
    """

    name = "searcher"
    tier = ModelTier.FAST        # only used for the optional query-rewrite call
    json_mode = True
    max_tokens = 300             # query rewrite needs ~100; a ReAct step (thought + action) a bit more

    async def run(self, state: StateLike) -> dict[str, Any]:
        sub_question = state.get("current_sub_question", "").strip()
        if not sub_question:
            raise SearcherError(
                "current_sub_question is missing or empty — "
                "the orchestrator must set it before dispatching the Searcher."
            )

        session_id = state["session_id"]
        iteration = state.get("iteration", 0)
        topic = state.get("query", "")

        self.log.info(
            f"searching for sub-question: {sub_question!r}",
            step="search_start",
            session_id=session_id,
            iteration=iteration,
        )

        pool: dict[str, SearchResult] = {}
        rankings: list[list[str]] = []
        prefetched: dict[str, str] = {}
        traces: list[dict] = []
        n_queries = 0
        if settings.search_mode == "react":
            outcome = await ReactSearch(self).run(sub_question, topic, state)
            pool, rankings, prefetched = outcome.pool, outcome.rankings, outcome.pages
            traces = [{"sub_question": sub_question, "stopped": outcome.stopped, "steps": outcome.trace}]
            n_queries = len(outcome.trace)
            if not pool:                                    # the loop found nothing (or broke): use the fixed recipe
                self.log.warning("react search produced no candidates; falling back to the fixed recipe",
                                 step="react_fallback", session_id=session_id, iteration=iteration)
        if not pool:
            pool, rankings, n_queries = await self._gather_fixed(sub_question, topic, state)

        fused = rrf_fuse(rankings)
        candidates = [r.model_copy(update={"url": r.url}) for r in pool.values()]
        by_key = {_url_key(r.url): r.url for r in candidates}
        fused_by_url = {by_key[k]: v for k, v in fused.items() if k in by_key}
        results = rank_and_diversify(
            candidates, fused_by_url, limit=settings.search_max_results,
            quality_weight=settings.source_quality_weight, max_per_domain=settings.max_results_per_domain,
        )

        if results and settings.fetch_pages:
            results = await self._enrich_with_pages(results, sub_question, session_id, iteration, prefetched)
        if results and settings.arxiv_fulltext:
            results = await self._enrich_with_arxiv_fulltext(results, sub_question, session_id, iteration)

        if not results:
            self.log.warning(
                f"zero results for sub-question: {sub_question!r}",
                step="search_empty",
                session_id=session_id,
                iteration=iteration,
            )
            return {"search_results": [], "failed_sub_questions": [sub_question], "react_traces": traces}

        self.log.info(
            f"found {len(results)} result(s) for sub-question: {sub_question!r} "
            f"({n_queries} query/ies, {sum(1 for r in results if r.source_type == 'arxiv')} arXiv)",
            step="search_done",
            session_id=session_id,
            iteration=iteration,
        )
        return {"search_results": results, "failed_sub_questions": [], "react_traces": traces}

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _gather_fixed(self, sub_question: str, topic: str, state: StateLike
                            ) -> tuple[dict[str, SearchResult], list[list[str]], int]:
        """The fixed recipe: verbatim + rewritten query on the web, plus arXiv.  Returns (pool, rankings, n_queries)."""
        queries = await self._queries(sub_question, topic, state)
        per_query = await asyncio.gather(*(
            search_web(q, sub_question=sub_question, max_results=settings.search_max_results) for q in queries
        ))
        extra = await self._arxiv(sub_question, topic)
        pool: dict[str, SearchResult] = {}
        rankings: list[list[str]] = []
        for res in [*per_query, extra]:
            urls = []
            for r in res:
                key = _url_key(r.url)
                pool.setdefault(key, r)
                urls.append(key)
            if urls:
                rankings.append(urls)
        return pool, rankings, len(queries)

    async def _queries(self, sub_question: str, topic: str, state: StateLike) -> list[str]:
        queries = [sub_question]
        extra_slots = max(0, settings.search_queries_per_subq - 1)
        mode = settings.query_rewrite
        if extra_slots == 0 or mode == "off":
            return queries
        extra: list[str] = []
        if mode == "llm":
            try:
                raw, _ = await self.llm_generate(
                    [{"role": "system", "content": REWRITE_SYSTEM},
                     {"role": "user", "content": rewrite_user_prompt(sub_question, topic)}],
                    state=state,
                )
                extra = parse_llm_queries(raw, sub_question, extra_slots)
            except BudgetExceeded:
                pass
            except Exception as exc:  # noqa: BLE001 — rewriting is an optimisation, never a blocker
                self.log.warning(f"query rewrite failed ({type(exc).__name__}); using rules",
                                 step="query_rewrite_failed", session_id=state["session_id"])
        if not extra:
            rq = rules_query(sub_question, topic)
            if rq and rq.lower() != sub_question.lower():
                extra = [rq]
        return queries + extra[:extra_slots]

    async def _arxiv(self, sub_question: str, topic: str) -> list[SearchResult]:
        if not settings.arxiv_enabled:
            return []
        found = await search_arxiv(
            rules_query(sub_question, topic, max_terms=6), sub_question=sub_question,
            max_results=settings.arxiv_max_results,
        )
        return [r for r in found if arxiv_relevant(sub_question, r, settings.arxiv_min_overlap)]

    async def _enrich_with_arxiv_fulltext(
        self, results: list[SearchResult], sub_question: str, session_id: str, iteration: int
    ) -> list[SearchResult]:
        """Replace the abstract with the best passages of the PAPER for the top-N arXiv results."""
        targets = [r.url for r in results if r.source_type == "arxiv"][: settings.arxiv_fulltext_top_n]
        texts: dict[str, str] = {}
        for url in targets:                         # sequential on purpose: arXiv asks for >= 3s between requests
            text = await fetch_pdf_text(url)
            if text:
                texts[url] = text
        out, n_ok = [], 0
        for r in results:
            text = texts.get(r.url)
            if text:
                passages = select_passages(sub_question, text, k=settings.fetch_passages_per_source,
                                           max_chars=settings.evidence_chars_per_source)
                if passages:
                    r = r.model_copy(update={"full_text": passages})
                    n_ok += 1
            out.append(r)
        self.log.info(
            f"arXiv full text for {n_ok}/{len(targets)} paper(s)",
            step="arxiv_fulltext",
            session_id=session_id,
            iteration=iteration,
        )
        return out

    async def _enrich_with_pages(
        self, results: list[SearchResult], sub_question: str, session_id: str, iteration: int,
        prefetched: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        top = [r for r in results[: settings.fetch_top_n] if r.source_type == "web"]
        have = prefetched or {}                                   # pages the ReAct loop already read
        pages = {**have, **await fetch_many(r.url for r in top if r.url not in have)}
        enriched = []
        n_ok = 0
        for r in results:
            text = pages.get(r.url)
            if text:
                passages = select_passages(
                    sub_question, text,
                    k=settings.fetch_passages_per_source,
                    max_chars=settings.evidence_chars_per_source,
                )
                if passages:
                    r = r.model_copy(update={"full_text": passages})
                    n_ok += 1
            enriched.append(r)
        self.log.info(
            f"fetched page text for {n_ok}/{len(top)} top result(s)",
            step="search_enriched",
            session_id=session_id,
            iteration=iteration,
        )
        return enriched


