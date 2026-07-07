"""
llm/client.py — INTERNAL provider orchestration layer

Do not import this module directly from agents or tools.
Use the package facade instead:

    import llm
    text = llm.generate("Explain AI")

    from llm import generate, ModelTier
    text = generate("Summarise this page", tier=ModelTier.FAST)
    text, usage = generate("Explain AI", return_usage=True)

Internally this module tries the primary provider, retries with backoff,
then falls back to the secondary provider.  The caller receives only the
text (and optionally a provider-agnostic usage dict) — never a raw SDK
object and never the name of whichever provider answered.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import List, Union

from groq import RateLimitError, APIStatusError
from google.genai.errors import APIError as GeminiAPIError, ClientError as GeminiClientError

from config import settings
from llm import groq_client, gemini_client

logger = logging.getLogger(__name__)

# ── retry / backoff constants (sourced from settings) ───────────────────────
_MAX_RETRIES: int = settings.llm_max_retries
_BACKOFF_BASE: float = settings.llm_backoff_base

# ── model tiers ─────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    """
    FAST   → cheap / quick model for high-frequency agents (Searcher, Reader)
    STRONG → capable model for reasoning agents (Synthesizer, Critic, Writer)
    """
    FAST = "fast"
    STRONG = "strong"


_GROQ_TIER_MAP = {
    ModelTier.FAST: groq_client.GROQ_FAST_MODEL,
    ModelTier.STRONG: groq_client.GROQ_STRONG_MODEL,
}
_GEMINI_TIER_MAP = {
    ModelTier.FAST: gemini_client.GEMINI_MODEL,
    ModelTier.STRONG: gemini_client.GEMINI_MODEL,
}


# ── public API ───────────────────────────────────────────────────────────────

def generate(
    prompt: Union[str, List[dict]],
    *,
    tier: ModelTier = ModelTier.STRONG,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """
    Generate a response from the LLM.

    Parameters
    ----------
    prompt      : plain string OR a list of {"role": ..., "content": ...} dicts
    tier        : ModelTier.FAST or ModelTier.STRONG
    system      : optional system message (ignored when prompt is already a list)
    temperature : sampling temperature (defaults to settings.llm_temperature)
    max_tokens  : maximum tokens to generate (defaults to settings.llm_max_tokens)
    return_usage: when True, returns (text, usage_dict) instead of just text

    Returns
    -------
    str if return_usage is False (default)
    tuple[str, dict] if return_usage is True
    """
    _temperature = temperature if temperature is not None else settings.llm_temperature
    _max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    messages = _build_messages(prompt, system)
    text, usage = _call_with_fallback(messages, tier=tier, temperature=_temperature, max_tokens=_max_tokens)

    if return_usage:
        return text, usage
    return text


# ── internals ────────────────────────────────────────────────────────────────

def _build_messages(prompt: Union[str, List[dict]], system: str | None) -> List[dict]:
    if isinstance(prompt, list):
        return prompt
    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _call_with_fallback(
    messages: List[dict],
    *,
    tier: ModelTier,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    groq_model = _GROQ_TIER_MAP[tier]
    gemini_model = _GEMINI_TIER_MAP[tier]

    # ── try Groq first (with retries on transient errors) ────────────────────
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return groq_client.generate(
                messages,
                model=groq_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning("Groq rate-limit (attempt %d/%d) — waiting %.1fs: %s", attempt, _MAX_RETRIES, wait, exc)
            time.sleep(wait)
        except APIStatusError as exc:
            logger.warning("Groq API error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            if attempt == _MAX_RETRIES:
                break
            time.sleep(_BACKOFF_BASE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq unexpected error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            break

    # ── primary exhausted — try secondary provider ───────────────────────────
    logger.info("Primary provider unavailable — trying secondary (%s)", gemini_model)
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return gemini_client.generate(
                messages,
                model=gemini_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except GeminiClientError as exc:
            if exc.code == 429:
                wait = _BACKOFF_BASE ** attempt
                logger.warning("Gemini rate-limit (attempt %d/%d) — waiting %.1fs: %s", attempt, _MAX_RETRIES, wait, exc)
                time.sleep(wait)
            else:
                logger.warning("Gemini client error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_BACKOFF_BASE)
        except GeminiAPIError as exc:
            logger.warning("Gemini API error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_BACKOFF_BASE)

    raise RuntimeError("All LLM providers failed after retries.")
