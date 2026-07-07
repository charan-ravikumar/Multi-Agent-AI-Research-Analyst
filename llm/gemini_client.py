from __future__ import annotations

import logging
import time
from typing import List

# ── New unified Google GenAI SDK (google-genai >= 2.x) ───────────────────────
# Replaces the deprecated google-generativeai SDK (deprecated 2025-11-30).
# API shape verified by introspecting google.genai 2.10.0:
#
#   client = genai.Client(api_key=...)
#   response = client.models.generate_content(
#       model=str,
#       contents=list[types.Content],
#       config=types.GenerateContentConfig(
#           system_instruction=str | None,
#           temperature=float | None,
#           max_output_tokens=int | None,
#       )
#   ) -> types.GenerateContentResponse
#   response.text          → str shortcut property
#   response.usage_metadata.prompt_token_count / candidates_token_count / total_token_count
#
# Exceptions raised by the new SDK:
#   google.genai.errors.ClientError (4xx, including 429 rate-limit)
#   google.genai.errors.ServerError  (5xx)
#   google.genai.errors.APIError     (base class for all the above)
# ─────────────────────────────────────────────────────────────────────────────
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ── INTERNAL MODULE — do not import from agents or tools ────────────────
# Use `import llm; llm.generate(...)` instead.

from config import settings

logger = logging.getLogger(__name__)

GEMINI_MODEL: str = settings.gemini_model

# Module-level client singleton — created once on first use.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def generate(
    messages: List[dict],
    *,
    model: str = GEMINI_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, dict]:
    """
    Call Gemini via the new google-genai SDK and return (response_text, usage_dict).
    Converts the OpenAI-style messages list to the new SDK's Content/Part format.
    Raises google.genai.errors.APIError so llm/client.py can handle retries.
    """
    client = _get_client()

    # Split system messages out of the conversation; convert user/assistant turns
    # to types.Content objects (role="user" or role="model").
    system_parts: list[str] = []
    contents: list[types.Content] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

    cfg = types.GenerateContentConfig(
        system_instruction="\n\n".join(system_parts) if system_parts else None,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    t0 = time.perf_counter()
    response = client.models.generate_content(model=model, contents=contents, config=cfg)
    latency = time.perf_counter() - t0

    text = response.text
    um = response.usage_metadata
    usage = {
        "provider": "gemini",
        "model": model,
        "prompt_tokens": um.prompt_token_count if um else None,
        "completion_tokens": um.candidates_token_count if um else None,
        "total_tokens": um.total_token_count if um else None,
        "latency_s": round(latency, 3),
    }
    logger.debug("Gemini response | %s", usage)
    return text, usage
    return text, usage
