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
    groq_api_key: str
    # Groq is the primary LLM provider. Gemini is the fallback via the new
    # google-genai SDK (2.x).  The old google-generativeai SDK was deprecated
    # on 2025-11-30 and caused an ImportError (missing grpc_asyncio transport)
    # whenever the fallback was triggered.  Migrated to google-genai==2.10.0;
    # confirmed working with gemini-2.5-flash.  GEMINI_API_KEY is required by
    # Pydantic validation at import time.
    gemini_api_key: str

    # ── LLM — model names ─────────────────────────────────────────────────────
    groq_fast_model: str = "llama-3.1-8b-instant"      # Searcher, Reader
    groq_strong_model: str = "llama-3.3-70b-versatile"  # Synthesizer, Critic, Writer
    gemini_model: str = "gemini-2.5-flash"               # fallback for all tiers
    gemini_judge_model: str = "gemini-2.5-pro"           # RAGAS judge

    # ── LLM — generation defaults ─────────────────────────────────────────────
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # ── LLM — retry / backoff ─────────────────────────────────────────────────
    llm_max_retries: int = 3
    llm_backoff_base: float = 2.0
    max_reflection_iterations: int = 3
    critic_score_threshold: float = 0.8
    # Maximum Critic-Synthesizer debate rounds before forced resolution.
    # Capped at 3 to bound LLM cost.  Same hard-boundary pattern as the
    # searcher zero-results retry cap.
    debate_max_rounds: int = 3

    # ── Redis (only memory layer) ─────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    cache_ttl_seconds: int = 3600

    # ── Web search / scraping ─────────────────────────────────────────────────
    search_max_results: int = 10
    search_timeout_seconds: int = 15
    scrape_max_concurrency: int = 4
    # ── Reader ────────────────────────────────────────────────────────────
    # Max number of concurrent LLM calls issued by the Reader when extracting
    # facts from multiple search results in parallel. Keeps Groq RPM in check.
    reader_max_concurrent_llm_calls: int = 3
    # ── Research planner ──────────────────────────────────────────────────────
    # Kept at 3 for the free-tier Groq account (6,000 TPM limit).
    # With N branches × 10 search results × 3 concurrent LLM calls per Reader,
    # each branch uses ~600-800 tokens/min. 3 branches = ~1800-2400 TPM, comfortably
    # within the 6,000 TPM limit. Raise to 5-10 when on a paid tier.
    planner_min_sub_questions: int = 3
    planner_max_sub_questions: int = 3

    # ── Academic sources ──────────────────────────────────────────────────────
    academic_max_results: int = 5

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/agent_traces.jsonl"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── General ───────────────────────────────────────────────────────────────
    environment: str = "development"


settings = Settings()


