"""
llm/gemini_client.py — INTERNAL: the only module that talks to the Gemini API (google-genai SDK).

Do not import from agents; use ``research_analyst.llm``.  Two entry points:

  generate()   text or STRUCTURED output: with ``schema`` the API constrains the reply to that Pydantic model
               (response_mime_type=application/json + response_schema), so the shape is guaranteed by the provider.
  tool_turn()  NATIVE FUNCTION CALLING: tools are declared to the API, ``mode=ANY`` forces the model to answer with a
               call to one of them (it cannot reply in prose), and automatic execution is disabled so the caller
               stays in control of what a tool actually does.

SDK errors (google.genai.errors.APIError, 4xx/5xx) propagate to llm/client.py, which owns retries and limits.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from research_analyst.config import settings
from research_analyst.llm.tools import ToolCall, ToolSpec, ToolTurn, to_declarations

logger = logging.getLogger(__name__)

_clients: dict[str, genai.Client] = {}          # one SDK client per API key


def _get_client(api_key: str | None = None) -> genai.Client:
    """The SDK client for ``api_key`` (default: the first configured key)."""
    key = api_key or settings.gemini_api_key
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = _clients.get(key)
    if client is None:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)))  # ms
        _clients[key] = client
    return client


def _thinking() -> types.ThinkingConfig | None:
    level = settings.gemini_thinking_level.strip().upper()
    return types.ThinkingConfig(thinking_level=level) if level else None   # type: ignore[arg-type]


def _split(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """OpenAI-style messages -> (system instruction, Gemini contents)."""
    system_parts: list[str] = []
    contents: list[types.Content] = []
    for msg in messages:
        role, text = msg["role"], msg["content"]
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
    return ("\n\n".join(system_parts) if system_parts else None), contents


def _finish_and_usage(response: Any, model: str, latency: float) -> dict:
    finish = None
    try:
        fr = (response.candidates or [])[0].finish_reason
        finish = getattr(fr, "name", str(fr)) if fr is not None else None
    except (AttributeError, IndexError, TypeError):
        pass
    if finish == "MAX_TOKENS":
        finish = "length"
    um = response.usage_metadata
    return {
        "provider": "gemini", "model": model,
        "prompt_tokens": um.prompt_token_count if um else None,
        "completion_tokens": um.candidates_token_count if um else None,
        "total_tokens": um.total_token_count if um else None,
        "latency_s": round(latency, 3), "finish_reason": finish,
    }


def generate(
    messages: list[dict], *, model: str, temperature: float = 0.2, max_tokens: int = 2048,
    json_mode: bool = False, schema: Any = None, on_text: Callable[[str], None] | None = None, api_key: str | None = None,
) -> tuple[str, dict]:
    """Text, or JSON constrained to ``schema`` (a Pydantic model class) when given.  Returns (text, usage).

    With ``on_text`` the answer is STREAMED: the callback receives the full text so far after every chunk (cumulative, so a
    retried attempt simply overwrites the previous one).  It is called from the calling thread and must be cheap."""
    system, contents = _split(messages)
    structured = schema is not None or json_mode
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, max_output_tokens=max_tokens,
        response_mime_type="application/json" if structured else None,
        response_schema=schema, thinking_config=_thinking(),
    )
    t0 = time.perf_counter()
    if on_text is not None:
        return _stream(model, contents, cfg, t0, on_text, api_key)
    response = _get_client(api_key).models.generate_content(model=model, contents=contents, config=cfg)  # type: ignore[arg-type]
    usage = _finish_and_usage(response, model, time.perf_counter() - t0)
    logger.debug("Gemini response | %s", usage)
    return response.text or "", usage


def _stream(model: str, contents: list[types.Content], cfg: types.GenerateContentConfig, t0: float,
            on_text: Callable[[str], None], api_key: str | None = None) -> tuple[str, dict]:
    on_text("")                                           # a retried attempt starts from an empty screen
    text, last = "", None
    for chunk in _get_client(api_key).models.generate_content_stream(model=model, contents=contents, config=cfg):  # type: ignore[arg-type]
        piece = chunk.text or ""
        if piece:
            text += piece
            try:
                on_text(text)
            except Exception:  # noqa: BLE001 — a broken display callback must not fail the generation
                logger.debug("on_text callback failed", exc_info=True)
        last = chunk
    usage = _finish_and_usage(last, model, time.perf_counter() - t0) if last is not None else {
        "provider": "gemini", "model": model, "prompt_tokens": None, "completion_tokens": None,
        "total_tokens": None, "latency_s": round(time.perf_counter() - t0, 3), "finish_reason": None}
    return text, usage


def tool_turn(
    system: str | None, history: list[Any], tools: list[ToolSpec], *, model: str,
    temperature: float = 0.2, max_tokens: int = 1024, api_key: str | None = None,
) -> ToolTurn:
    """One model turn in a tool-calling conversation.  ``history`` is a list of provider Content objects."""
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, max_output_tokens=max_tokens,
        tools=to_declarations(tools),  # type: ignore[arg-type]
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=[t.name for t in tools])),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=_thinking(),
    )
    t0 = time.perf_counter()
    response = _get_client(api_key).models.generate_content(model=model, contents=history, config=cfg)
    usage = _finish_and_usage(response, model, time.perf_counter() - t0)
    cands = response.candidates or []
    content = cands[0].content if cands else None
    parts = (content.parts or []) if content else []
    calls = [ToolCall(name=p.function_call.name or "", args=dict(p.function_call.args or {}), id=p.function_call.id)
             for p in parts if p.function_call]
    text = "".join(p.text or "" for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    return ToolTurn(calls=calls, text=text, content=content, usage=usage)
