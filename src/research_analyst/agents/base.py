"""
agents/base.py — abstract base class for all research agents

Every agent in this project:
  1. Inherits from BaseAgent
  2. Declares a class-level `name` and optionally `tier`
  3. Implements `async def run(self, state: StateLike) -> dict[str, Any]`

The base class __call__ wraps run() with:
  - wall-clock timing
  - structured logging (start + completion)
  - consistent exception handling

LangGraph registers each agent as a graph node via its __call__:

    from research_analyst.agents.planner import PlannerAgent
    graph.add_node("planner", PlannerAgent())
"""
from __future__ import annotations

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Mapping

from research_analyst import llm
from research_analyst.observability import tracing
from research_analyst.observability.logger import get_logger
from research_analyst.observability.usage import estimate_cost, tracker
from research_analyst.llm import ModelTier


# A node receives the (possibly branch-local) graph state and returns a PARTIAL update.
StateLike = Mapping[str, Any]


class BaseAgent(ABC):
    """
    Abstract base for all six research agents.

    Subclasses must set:
        name : str       — unique snake_case identifier, used in logs and Redis keys
        tier : ModelTier — which LLM tier to use (default: STRONG)

    Subclasses must implement:
        async def run(self, state: StateLike) -> dict[str, Any]
    """

    name: str = "base"
    tier: ModelTier = ModelTier.STRONG
    json_mode: bool = False          # ask the provider for a JSON object
    max_tokens: int | None = None    # None -> settings.llm_max_tokens
    prefer_provider: str | None = None   # e.g. "gemini" for the Critic
    groq_model: str | None = None        # per-agent Groq model override (None = tier default)
    budget_exempt: bool = False          # True: never blocked by settings.token_budget_per_run (Writer)

    def __init__(self) -> None:
        self.log = get_logger(self.name)

    # ── public interface ──────────────────────────────────────────────────────

    async def __call__(self, state: StateLike) -> dict[str, Any]:
        """
        LangGraph node entry-point.

        Wraps run() with timing, logging, and Redis persistence so subclasses
        contain only domain logic and never repeat instrumentation boilerplate.
        """
        session_id = state["session_id"]
        iteration = state.get("iteration", 0)

        self.log.info(
            f"{self.name} starting",
            step="agent_start",
            session_id=session_id,
            iteration=iteration,
        )

        t0 = time.perf_counter()
        with tracing.observation(
            f"agent:{self.name}", as_type="agent", session_id=session_id,
            input={"sub_question": state.get("current_sub_question") or state.get("query", "")},
        ):
            try:
                result = await self.run(state)
            except Exception as exc:
                latency = time.perf_counter() - t0
                self.log.error(
                    f"{self.name} failed: {exc}",
                    step="agent_error",
                    session_id=session_id,
                    iteration=iteration,
                    latency_s=round(latency, 3),
                    exc_traceback=traceback.format_exc(),
                )
                raise

        latency = time.perf_counter() - t0

        self.log.log_call(
            "agent_complete",
            session_id=session_id,
            iteration=iteration,
            latency_s=round(latency, 3),
            message=f"{self.name} completed in {latency:.2f}s",
        )

        return result

    @abstractmethod
    async def run(self, state: StateLike) -> dict[str, Any]:
        """
        Domain logic for this agent.

        Read from state, do work, return an updated state dict.
        Never call __call__ recursively.
        """
        ...

    # ── helpers available to all subclasses ───────────────────────────────────

    async def llm_generate(
        self,
        prompt: str | list,
        *,
        system: str | None = None,
        state: StateLike,
    ) -> tuple[str, dict]:
        """
        Call the LLM with this agent's tier and auto-log the result.
        Runs the blocking llm.generate() in a thread so it doesn't stall
        the event loop when multiple agents execute concurrently.
        Returns (text, usage) — usage is passed straight to log_call.
        """
        session_id = state["session_id"]
        if not self.budget_exempt:
            tracker.check(session_id)          # raises BudgetExceeded when the run's budget is spent
        with tracing.observation(
            f"llm:{self.name}", as_type="generation", session_id=session_id, input=prompt,
            model_parameters={"json_mode": self.json_mode, "max_tokens": self.max_tokens or 0,
                              "tier": self.tier.value},
        ) as gen:
            text, usage = await asyncio.to_thread(
                llm.generate,
                prompt,
                tier=self.tier,
                system=system,
                return_usage=True,
                json_mode=self.json_mode,
                max_tokens=self.max_tokens,
                prefer=self.prefer_provider,
                groq_model=self.groq_model,
            )
            cost = estimate_cost(usage)
            gen.update(
                output=text,
                model=usage.get("model"),
                usage_details={k: v for k, v in {
                    "input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens"),
                    "total": usage.get("total_tokens")}.items() if v is not None},
                cost_details={"total": cost} if cost is not None else None,
                metadata={"provider": usage.get("provider"), "finish_reason": usage.get("finish_reason"),
                          "latency_s": usage.get("latency_s")},
            )
        tracker.record(session_id, self.name, usage)
        if usage.get("finish_reason") == "length":
            self.log.warning(
                f"{self.name}: output hit max_tokens and was truncated",
                step="llm_truncated",
                session_id=state["session_id"],
                iteration=state.get("iteration", 0),
            )
        self.log.log_call(
            "llm_call",
            session_id=state["session_id"],
            iteration=state.get("iteration", 0),
            latency_s=usage.get("latency_s"),
            tokens=usage,
        )
        return text, usage
