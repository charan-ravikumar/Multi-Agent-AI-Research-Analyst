"""
agents/base.py — abstract base class for all research agents

Every agent in this project:
  1. Inherits from BaseAgent
  2. Declares a class-level `name` and optionally `tier`
  3. Implements `async def run(self, state: ResearchState) -> ResearchState`

The base class __call__ wraps run() with:
  - wall-clock timing
  - structured logging (start + completion)
  - Redis scratchpad persistence
  - consistent exception handling

LangGraph registers each agent as a graph node via its __call__:

    from agents.planner import PlannerAgent
    graph.add_node("planner", PlannerAgent())
"""
from __future__ import annotations

import asyncio
import time
import traceback
from abc import ABC, abstractmethod

import llm
from core.logger import get_logger
from memory.redis_store import save_scratchpad
from models import ResearchState
from llm import ModelTier


class BaseAgent(ABC):
    """
    Abstract base for all six research agents.

    Subclasses must set:
        name : str       — unique snake_case identifier, used in logs and Redis keys
        tier : ModelTier — which LLM tier to use (default: STRONG)

    Subclasses must implement:
        async def run(self, state: ResearchState) -> ResearchState
    """

    name: str = "base"
    tier: ModelTier = ModelTier.STRONG

    def __init__(self) -> None:
        self.log = get_logger(self.name)

    # ── public interface ──────────────────────────────────────────────────────

    async def __call__(self, state: ResearchState) -> ResearchState:
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

        # persist scratchpad so the Streamlit UI and Redis inspector can see
        # what each agent last wrote without re-running the pipeline
        self._persist_scratchpad(result)

        return result

    @abstractmethod
    async def run(self, state: ResearchState) -> ResearchState:
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
        state: ResearchState,
    ) -> tuple[str, dict]:
        """
        Call the LLM with this agent's tier and auto-log the result.
        Runs the blocking llm.generate() in a thread so it doesn't stall
        the event loop when multiple agents execute concurrently.
        Returns (text, usage) — usage is passed straight to log_call.
        """
        text, usage = await asyncio.to_thread(
            llm.generate,
            prompt,
            tier=self.tier,
            system=system,
            return_usage=True,
        )
        self.log.log_call(
            "llm_call",
            session_id=state["session_id"],
            iteration=state.get("iteration", 0),
            latency_s=usage.get("latency_s"),
            tokens=usage,
        )
        return text, usage

    # ── internals ─────────────────────────────────────────────────────────────

    def _persist_scratchpad(self, state: ResearchState) -> None:
        """Mirror key state fields to Redis so they survive across restarts."""
        session_id = state.get("session_id", "unknown")
        try:
            save_scratchpad(
                session_id,
                self.name,
                {
                    "iteration": state.get("iteration", 0),
                    "sub_questions": state.get("sub_questions", []),
                    "search_result_count": len(state.get("search_results", [])),
                    "fact_count": len(state.get("extracted_facts", [])),
                    "has_draft": bool(state.get("draft")),
                    "has_final_report": bool(state.get("final_report")),
                },
            )
        except Exception:
            # Redis persistence is best-effort — never crash the pipeline
            self.log.warning(
                "scratchpad persist failed",
                step="scratchpad",
                session_id=session_id,
                iteration=state.get("iteration", 0),
            )
