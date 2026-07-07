from __future__ import annotations

import logging
import time
from typing import List

from groq import Groq, RateLimitError, APIStatusError

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
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def generate(
    messages: List[dict],
    *,
    model: str = GROQ_STRONG_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, dict]:
    """
    Call Groq and return (response_text, usage_dict).
    Raises RateLimitError or APIStatusError so the caller can fall back.
    """
    client = _get_client()
    t0 = time.perf_counter()

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    latency = time.perf_counter() - t0
    usage = {
        "provider": "groq",
        "model": model,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
        "latency_s": round(latency, 3),
    }
    text = completion.choices[0].message.content
    logger.debug("Groq response | %s", usage)
    return text, usage
