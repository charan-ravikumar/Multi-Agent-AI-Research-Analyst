from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single source of truth for every configurable value in the project.
    Loaded from .env (or real environment variables).
    Nothing else in the codebase should call os.getenv().

    Vector-store / Chroma / embeddings are intentionally absent — this project
    does NOT do retrieval-via-embeddings.  Memory is Redis-only.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM — Gemini only (no fallback provider) ──────────────────────────────
    # Everything goes through the Gemini API.  A failing call is retried (429/5xx) and then raises; there is no
    # second provider to hide an outage.  Keys default to "" so the package imports without credentials.
    gemini_api_key: str = ""
    # Two models: a cheap high-volume one for most agents, a stronger one only where reasoning quality matters.
    gemini_fast_model: str = "gemini-3.5-flash-lite"     # 15 RPM, 250K TPM, 500 RPD on the free tier
    gemini_strong_model: str = "gemini-3.8-flash"        # 5 RPM, 250K TPM, 20 RPD on the free tier
    # Comma-separated agent names that use the STRONG model.  Every other agent uses the fast model.  The strong model
    # has only 20 requests/day, so the default is the Writer alone (1-2 calls per run).
    gemini_strong_agents: str = "writer"
    # Model for the LLM judges (grounding judge + evaluation judge); empty = the fast model.
    judge_model: str = ""
    # Free-tier limits per model, JSON: {"model": {"rpm": n, "tpm": n, "rpd": n}}.  Calls wait for RPM/TPM room and
    # fail fast with a clear error once RPD is spent (instead of burning retries on 429s).  {} disables limiting.
    gemini_rate_limits: str = (
        '{"gemini-3.8-flash": {"rpm": 5, "tpm": 250000, "rpd": 20}, '
        '"gemini-3.5-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500}}'
    )
    # Where the per-day request counter is kept (resets at midnight Pacific, like Google's quota).
    llm_quota_file: str = "logs/llm_quota.json"
    # Gemini 3 models take a thinking LEVEL (thinking_budget=0 is rejected): "", "minimal", "low", "medium", "high".
    # Empty = the model default.  Thinking tokens count against max_tokens, so lower it if answers get truncated.
    gemini_thinking_level: str = ""

    # ── LLM — generation defaults ─────────────────────────────────────────────
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # ── LLM — retry / backoff ─────────────────────────────────────────────────
    llm_max_retries: int = 3
    llm_backoff_base: float = 2.0
    # One retry policy lives in llm/client.py (SDK-level retries are disabled so
    # attempts do not multiply).  Delays use jitter and honour Retry-After.
    llm_timeout_seconds: float = 45.0          # per HTTP request, both providers
    llm_retry_after_cap_seconds: float = 20.0  # never sleep longer than this on Retry-After
    # Circuit breaker: after N consecutive exhausted calls a provider is skipped
    # for the cooldown instead of making every call pay the full retry cost.
    llm_breaker_threshold: int = 3
    llm_breaker_cooldown_seconds: float = 60.0
    # ── Per-run accounting ─────────────────────────────────────────────────────
    # Hard stop for LLM spend per research run (prompt+completion tokens, summed over
    # all calls).  0 disables.  When hit: branches fail (isolated), the debate stops
    # early (disclosed in the report); the Writer is exempt so a report is still produced.
    token_budget_per_run: int = 0
    # Optional price table for cost estimates, JSON: {"model-name": [usd_per_1M_input, usd_per_1M_output]}.
    # Empty = no cost estimate (the free tiers cost $0; tokens are still counted).
    llm_price_per_1m_tokens: str = ""
    # ── Input safety ───────────────────────────────────────────────────────────
    max_query_chars: int = 500
    # Maximum Critic-Synthesizer debate rounds before forced resolution.
    # Capped at 3 to bound LLM cost.  Same hard-boundary pattern as the
    # searcher zero-results retry cap.
    # Default 0 = review loop OFF.  Measured on the golden set (eval/RESULTS.md): the loop adds
    # citation markers but not faithfulness, lowers completeness and costs ~75% more tokens, so it
    # is opt-in (per run via the UI checkbox -> state["review_rounds"], or DEBATE_MAX_ROUNDS=2).
    # Note the last round only critiques (to disclose issues): 2 rounds = 1 revision.
    debate_max_rounds: int = 0
    # Wall-clock deadline for one research run (seconds); enforced by the UI/demo.
    run_timeout_seconds: int = 900

    # ── Redis (only memory layer) ─────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── Web search / scraping ─────────────────────────────────────────────────
    search_max_results: int = 10
    # ── Reader ────────────────────────────────────────────────────────────
    # Max number of concurrent LLM calls issued by the Reader when extracting
    # facts from multiple search results in parallel. Keeps the RPM limit in check.
    reader_max_concurrent_llm_calls: int = 3
    # Max sources sent to the (batched) Reader per sub-question and the evidence
    # text budget per source, to stay inside free-tier TPM limits.
    reader_max_sources: int = 9          # hard cap on sources read per sub-question
    reader_batch_size: int = 3           # sources per LLM call; reading stops early (see agents/reader.py)
    reader_target_facts: int = 6         # stop once this many distinct facts are collected
    reader_min_new_facts: int = 2        # stop when a later batch adds fewer new facts than this
    evidence_chars_per_source: int = 1200
    evidence_block_max_chars: int = 6000   # evidence list shown to the Critic/reviser
    # ── Page fetching (evidence beyond search-result snippets) ─────────────────
    fetch_pages: bool = True
    fetch_top_n: int = 4                 # pages fetched per sub-question
    fetch_max_concurrency: int = 4
    fetch_timeout_seconds: float = 8.0
    fetch_max_bytes: int = 1_500_000
    fetch_passages_per_source: int = 3   # BM25 top-k passages kept per page
    # ── Evidence quality ───────────────────────────────────────────────────────
    # "llm": one tiny FAST-model call per sub-question writes a keyword-style second search query;
    # "rules": free keyword extraction; "off": the sub-question is the only query.
    query_rewrite: str = "llm"
    search_queries_per_subq: int = 2         # verbatim sub-question + up to N-1 rewritten queries
    source_quality_weight: float = 0.5       # blend of heuristic domain quality into the ranking
    max_results_per_domain: int = 2          # diversity: cap per registered domain among kept results
    arxiv_enabled: bool = True               # add arXiv abstracts when clearly relevant to the sub-question
    arxiv_max_results: int = 3
    arxiv_min_overlap: int = 3               # shared key terms (sub-question vs title+abstract) required
    arxiv_min_interval_seconds: float = 3.0  # arXiv API etiquette: >= 3s between requests
    # Full text: download the PDF of the top-N relevant arXiv papers, extract text (pypdf) and keep the BM25
    # best passages for the sub-question — instead of the abstract alone.
    arxiv_fulltext: bool = True
    arxiv_fulltext_top_n: int = 2
    arxiv_pdf_max_bytes: int = 8_000_000
    arxiv_pdf_max_pages: int = 15
    # ── Search strategy ────────────────────────────────────────────────────────
    # "fixed": the same recipe for every sub-question (two phrasings + arXiv).
    # "react": a Reason+Act loop (agents/react_search.py) chooses searches / page reads from what it has seen,
    #          then the SAME ranking + fetching stage runs.  Costs ~1 small LLM call per step.
    search_mode: str = "fixed"
    react_max_steps: int = 4
    # ── Grounding judge (LLM) ──────────────────────────────────────────────────
    # After the report is written, an LLM checks each cited claim against the passages of the sources it cites
    # (supported / partial / unsupported).  Replaces the lexical label in the verify panel; the lexical score
    # is kept only to pick WHICH passage to show.  Falls back to lexical, disclosed, if the judge cannot run.
    grounding_judge: bool = True
    grounding_judge_max_claims: int = 24
    grounding_judge_batch_size: int = 6
    # ── Research planner ──────────────────────────────────────────────────────
    # Number of sub-questions the Planner must produce.  Each sub-question costs
    # ~reader results + 1 synthesis call, so on the free tier (15 RPM / 500 RPD for the fast model) raise these
    # with care: every sub-question multiplies the number of LLM calls per run.
    planner_min_sub_questions: int = 4
    planner_max_sub_questions: int = 5

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/agent_traces.jsonl"
    # One JSON line per finished research run (status, time, tokens, quality) — feeds the Monitor page.
    runs_file: str = "logs/runs.jsonl"
    # "json": one JSON object per line on stderr (for log shippers / Streamlit Cloud logs); "text": readable.
    log_format: str = "text"
    # Langfuse tracing (optional): enabled only when BOTH keys are set.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    # Langfuse's own docs/site use LANGFUSE_BASE_URL (region-specific: cloud / us.cloud / jp.cloud ...).
    # It takes precedence over LANGFUSE_HOST when set.
    langfuse_base_url: str = ""

    @property
    def strong_agents(self) -> frozenset[str]:
        """Names of the agents that use the STRONG (reasoning) model; everyone else uses the fast model."""
        return frozenset(a.strip() for a in self.gemini_strong_agents.split(",") if a.strip())

    @property
    def langfuse_url(self) -> str:
        return (self.langfuse_base_url or self.langfuse_host).rstrip("/")
    # Send prompts/completions to Langfuse (truncated).  Disable to send metadata + usage only.
    # Prompt management: take SYSTEM prompts from Langfuse (label below) instead of the in-repo text.  Active only when
    # tracing is enabled; any problem falls back to the in-repo prompt.  Push prompts with scripts/sync_prompts.py.
    langfuse_prompts: bool = True
    langfuse_prompt_label: str = "production"      # e.g. "staging" to evaluate a new version without touching production
    langfuse_prompt_cache_seconds: int = 300
    langfuse_capture_io: bool = True
    langfuse_max_io_chars: int = 4000

    # ── General ───────────────────────────────────────────────────────────────
    environment: str = "development"


settings = Settings()


