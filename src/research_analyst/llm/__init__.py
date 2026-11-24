"""
llm — public LLM interface for agents (Gemini only, no fallback provider)

Agents import ONLY from this package:

    from research_analyst import llm
    text = llm.generate("Explain AI")
    turn = llm.tool_turn(system, history, tools)          # native function calling

Agents must never import from llm.gemini_client or llm.client directly.
"""
from research_analyst.llm.client import DailyQuotaExceeded, ModelTier, generate, model_for, tool_turn
from research_analyst.llm.tools import ToolCall, ToolSpec, ToolTurn, tool_results_turn, user_turn

__all__ = ["generate", "tool_turn", "ModelTier", "DailyQuotaExceeded", "model_for",
           "ToolSpec", "ToolCall", "ToolTurn", "user_turn", "tool_results_turn"]
