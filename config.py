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

    # ── LLM — API keys ────────────────────────────────────────────────────────
    # Keys default to "" so the package imports (tests, CI) without credentials;
    # the provider clients raise a clear error on first real use if one is missing.
    groq_api_key: str = ""
    # Groq is the primary LLM provider. Gemini is the fallback via the new
    # google-genai SDK (2.x).  The old google-generativeai SDK was deprecated
    # on 2025-11-30 and caused an ImportError (missing grpc_asyncio transport)
    # whenever the fallback was triggered.  Migrated to google-genai==2.10.0;
    # confirmed working with gemini-2.5-flash.  Keys are validated lazily at first call.
    gemini_api_key: str = ""

    # ── LLM — model names ─────────────────────────────────────────────────────
    groq_fast_model: str = "openai/gpt-oss-20b"         # Reader (llama-3.1-8b-instant was retired for this key)
    groq_strong_model: str = "openai/gpt-oss-120b"      # Planner, Synthesizer, Critic, Writer
    # gpt-oss models are reasoning models: reasoning tokens count against max_tokens and TPM.
    # "low" keeps extraction/synthesis cheap; applied only to openai/gpt-oss-* models.
    groq_reasoning_effort: str = "low"
    # Second Groq model tried (own quota) when the primary model fails with an HTTP status error
    # (429/404/5xx) BEFORE falling back to Gemini, whose free tier is only ~20 requests/day.
    # Empty disables.  Different model family from the primary, so it also diversifies failures.
    groq_fallback_model: str = "qwen/qwen3.8-27b"
    # Independent judge for evaluation and optional model for the Critic (empty = tier default).
    judge_model: str = "qwen/qwen3.8-27b"
    critic_model: str = ""
    # Gemini 2.5 Flash is a "thinking" model: hidden reasoning tokens count against max_output_tokens
    # and truncated JSON answers in live tests.  0 disables thinking (Flash supports it).
    gemini_thinking_budget: int = 0
    gemini_model: str = "gemini-2.5-flash"               # fallback for all tiers

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
    # Sliding-window tokens-per-minute budget per model (prompt estimate + max_tokens);
    # 0 disables.  Set just under your provider TPM limit (Groq free 8B: 6000).
    llm_tpm_budget: int = 0
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
    # Run the Critic on Gemini first (different model family from the Groq
    # Synthesizer -> fewer shared blind spots); falls back to Groq.
    critic_prefer_gemini: bool = False
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
    # facts from multiple search results in parallel. Keeps Groq RPM in check.
    reader_max_concurrent_llm_calls: int = 3
    # Max sources sent to the (batched) Reader per sub-question and the evidence
    # text budget per source, to stay inside free-tier TPM limits.
    reader_max_sources: int = 6
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
    # ── Research planner ──────────────────────────────────────────────────────
    # Number of sub-questions the Planner must produce.  Each sub-question costs
    # ~reader results + 1 synthesis call, so on the Groq free tier (6,000 TPM on
    # the 8B model) raise these with care: real runs hit 429s at 5 x 5 results.
    planner_min_sub_questions: int = 4
    planner_max_sub_questions: int = 5

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/agent_traces.jsonl"
    # Langfuse tracing (optional): enabled only when BOTH keys are set.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    # Langfuse's own docs/site use LANGFUSE_BASE_URL (region-specific: cloud / us.cloud / jp.cloud ...).
    # It takes precedence over LANGFUSE_HOST when set.
    langfuse_base_url: str = ""

    @property
    def langfuse_url(self) -> str:
        return (self.langfuse_base_url or self.langfuse_host).rstrip("/")
    # Send prompts/completions to Langfuse (truncated).  Disable to send metadata + usage only.
    langfuse_capture_io: bool = True
    langfuse_max_io_chars: int = 4000

    # ── General ───────────────────────────────────────────────────────────────
    environment: str = "development"


settings = Settings()


