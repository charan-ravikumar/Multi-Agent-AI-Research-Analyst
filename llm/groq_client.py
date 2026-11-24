from __future__ import annotations

import logging
import time
from typing import List

from groq import Groq

# ── INTERNAL MODULE — do not import from agents or tools ─────────────────────
# Use `import llm; llm.generate(...)` instead.

from config import settings

logger = logging.getLogger(__name__)

GROQ_FAST_MODEL: str = settings.groq_fast_model
GROQ_STRONG_MODEL: str = settings.groq_strong_model

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        # max_retries=0: the single retry policy lives in llm/client.py
        _client = Groq(api_key=settings.groq_api_key, timeout=settings.llm_timeout_seconds, max_retries=0)
    return _client


def generate(
    messages: List[dict],
    *,
    model: str = GROQ_STRONG_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> tuple[str, dict]:
    """
    Call Groq and return (response_text, usage_dict).
    Raises RateLimitError or APIStatusError so the caller can fall back.
    """
    client = _get_client()
    t0 = time.perf_counter()

    kwargs: dict = {}
    if model.startswith("openai/gpt-oss") and settings.groq_reasoning_effort:
        kwargs["reasoning_effort"] = settings.groq_reasoning_effort
    if json_mode:
        # Groq JSON mode: guarantees a syntactically valid JSON *object*
        # (the prompt must mention JSON; every JSON stage here does).
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]  # plain role/content dicts are what the API takes
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    latency = time.perf_counter() - t0
    usage = {
        "provider": "groq",
        "model": model,
        "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
        "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
        "total_tokens": completion.usage.total_tokens if completion.usage else None,
        "latency_s": round(latency, 3),
        "finish_reason": completion.choices[0].finish_reason,   # "length" == truncated
    }
    text = completion.choices[0].message.content or ""
    logger.debug("Groq response | %s", usage)
    return text, usage
