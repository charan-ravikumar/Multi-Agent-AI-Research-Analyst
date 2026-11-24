"""
llm/tools.py — provider-neutral types for native function calling.

A tool is declared once as a ``ToolSpec`` (name, description, JSON-schema parameters).  The model answers a turn with
``ToolCall``s; the caller executes them and sends the results back.  The conversation history is opaque to callers:
it is whatever the provider needs to continue the exchange (for Gemini 3 that includes thought signatures, so the
model's own turn must be passed back untouched) — build it only with ``user_turn`` / ``tool_results_turn`` and append
``ToolTurn.content``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from google.genai import types


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema, type "object"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class ToolTurn:
    calls: list[ToolCall]
    text: str = ""                      # any prose the model added next to the calls
    content: Any = None                 # the model's turn in provider format: append it to the history unchanged
    usage: dict = field(default_factory=dict)


def user_turn(text: str) -> Any:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def tool_results_turn(results: list[tuple[ToolCall, dict[str, Any]]]) -> Any:
    """One user turn answering every call of the previous model turn (the API requires a response per call)."""
    return types.Content(role="user", parts=[
        types.Part(function_response=types.FunctionResponse(name=c.name, id=c.id, response=r)) for c, r in results])


def to_declarations(tools: list[ToolSpec]) -> list[types.Tool]:
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
        for t in tools])]
