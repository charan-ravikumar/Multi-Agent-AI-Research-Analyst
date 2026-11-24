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
import hashlib
import contextlib
import json
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar
from collections.abc import Mapping

from research_analyst import llm
from research_analyst.config import settings
from research_analyst.observability import metrics, tracing
from research_analyst.observability.logger import get_logger
from research_analyst.observability import prompt_store
from research_analyst.observability.usage import estimate_cost, tracker
from research_analyst.llm import ModelTier

T = TypeVar("T")
_DEFAULT: Any = object()   # sentinel: "use the agent's response_schema"

# A node receives the (possibly branch-local) graph state and returns a PARTIAL update.
StateLike = Mapping[str, Any]


class BaseAgent(ABC):
    """
    Abstract base for all six research agents.

    Subclasses must set:
        name : str       — unique snake_case identifier, used in logs and Redis keys

    The model tier is NOT declared by subclasses: ``tier`` is STRONG only for agents listed in
    settings.gemini_strong_agents, FAST otherwise (see llm/client.py for what each tier maps to).

    Subclasses must implement:
        async def run(self, state: StateLike) -> dict[str, Any]
    """

    name: str = "base"
    json_mode: bool = False          # ask the provider for valid JSON (use response_schema for a guaranteed SHAPE)
    response_schema: Any = None      # Pydantic model class: structured output enforced by the API
    max_tokens: int | None = None    # None -> settings.llm_max_tokens
    model: str | None = None         # explicit model for this agent (None = chosen by tier)
    budget_exempt: bool = False          # True: never blocked by settings.token_budget_per_run (Writer)

    def __init__(self) -> None:
        self.log = get_logger(self.name)

    @property
    def tier(self) -> ModelTier:
        return ModelTier.STRONG if self.name in settings.strong_agents else ModelTier.FAST

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
        schema: Any = _DEFAULT,
        on_text: Callable[[str], None] | None = None,
    ) -> tuple[str, dict]:
        """
        Call the LLM with this agent's tier and auto-log the result.  ``schema`` (a Pydantic model class) makes the
        API return JSON of exactly that shape; it defaults to the agent's ``response_schema``, pass None to opt out.
        Runs the blocking llm.generate() in a thread so it doesn't stall the event loop.
        Returns (text, usage).
        """
        schema = self.response_schema if schema is _DEFAULT else schema
        sys_text = next((m["content"] for m in prompt if m.get("role") == "system"), "") if isinstance(prompt, list) else (system or "")

        def call() -> tuple[str, dict]:
            extra = {"on_text": on_text} if on_text is not None else {}      # only the streaming caller passes it
            return llm.generate(prompt, tier=self.tier, system=system, return_usage=True,  # type: ignore[call-overload]
                                json_mode=self.json_mode or schema is not None, schema=schema,
                                max_tokens=self.max_tokens, model=self.model, **extra)

        return await self._instrumented(call, state=state, input_=prompt, sys_text=sys_text,
                                        params={"json_mode": self.json_mode, "schema": getattr(schema, "__name__", None)})

    async def llm_tool_turn(
        self, history: list, tools: list[llm.ToolSpec], *, system: str, state: StateLike,
    ) -> llm.ToolTurn:
        """One native function-calling turn (the model must answer with a call to one of ``tools``), fully traced."""
        def call() -> tuple[llm.ToolTurn, dict]:
            turn = llm.tool_turn(system, history, tools, tier=self.tier, max_tokens=self.max_tokens, model=self.model)
            return turn, turn.usage

        turn, _usage = await self._instrumented(
            call, state=state, input_=[{"role": "system", "content": system}], sys_text=system,
            params={"tools": [t.name for t in tools]},
            output_of=lambda t: json.dumps([{"name": c.name, "args": c.args} for c in t.calls]) + (f" | {t.text}" if t.text else ""))
        return turn

    async def _instrumented(
        self, call: Callable[[], tuple[Any, dict]], *, state: StateLike, input_: Any, sys_text: str,
        params: dict, output_of: Callable[[Any], str] = str,
    ) -> tuple[Any, dict]:
        """Budget check -> run ``call`` in a thread -> Langfuse generation, usage accounting, truncation warning."""
        session_id = state["session_id"]
        if not self.budget_exempt:
            tracker.check(session_id)          # raises BudgetExceeded when the run's budget is spent
        # A short hash of the system prompt identifies WHICH prompt text produced a trace, so a quality change
        # in Langfuse can be tied to a prompt edit (shown as the generation's "version").
        prompt_version = hashlib.sha1(sys_text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8] if sys_text else None
        managed = prompt_store.lookup(sys_text) if sys_text else None      # Langfuse-managed prompt this text came from
        link = {"prompt": managed.lf} if managed is not None and managed.lf is not None else {}
        with tracing.observation(
            f"llm:{self.name}", as_type="generation", session_id=session_id, input=input_, version=prompt_version, **link,
            model_parameters={**params, "max_tokens": self.max_tokens or 0, "tier": self.tier.value},
        ) as gen:
            result, usage = await asyncio.to_thread(call)
            cost = estimate_cost(usage)
            gen.update(
                output=output_of(result),
                model=usage.get("model"),
                usage_details={k: v for k, v in {
                    "input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens"),
                    "total": usage.get("total_tokens")}.items() if v is not None},
                cost_details={"total": cost} if cost is not None else None,
                metadata={"agent": self.name, "prompt_version": prompt_version,
                          "prompt_name": managed.name if managed else None,
                          "prompt_source": managed.source if managed else None,
                          "prompt_managed_version": managed.version if managed else None,
                          "provider": usage.get("provider"), "finish_reason": usage.get("finish_reason"),
                          "latency_s": usage.get("latency_s")},
            )
        tracker.record(session_id, self.name, usage)
        if usage.get("finish_reason") == "length":
            self.log.warning(
                f"{self.name}: output hit max_tokens and was truncated",
                step="llm_truncated",
                session_id=session_id,
                iteration=state.get("iteration", 0),
            )
        self.log.log_call(
            "llm_call",
            session_id=session_id,
            iteration=state.get("iteration", 0),
            latency_s=usage.get("latency_s"),
            tokens=usage,
        )
        return result, usage

    async def generate_parsed(
        self,
        messages: list[dict],
        parse: Callable[[str], T],
        state: StateLike,
        *,
        retry_message: str,
        step: str,
        attempts: int = 2,
        guard: Any = None,
        schema: Any = _DEFAULT,
    ) -> tuple[T | None, str]:
        """Generate -> parse; on a parse failure, show the model its own answer plus the error and retry.

        The one place that implements the "structured output with a corrective retry" pattern.  ``parse``
        must raise ``ValueError`` (``json.JSONDecodeError`` and pydantic's ``ValidationError`` both are)
        for an unusable answer.  ``guard`` is an optional async context manager (e.g. a semaphore) held
        around each LLM call.  Returns ``(value, last_error)``; ``value`` is None when every attempt failed.
        Transport errors (RuntimeError from the LLM layer) propagate: they are not parse failures.
        """
        last_raw, last_error = "", ""
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append({"role": "user", "content": retry_message.format(error=last_error)})
            async with guard or contextlib.nullcontext():
                last_raw, _usage = await self.llm_generate(messages, state=state, schema=schema)
            try:
                return parse(last_raw), ""
            except ValueError as exc:
                last_error = str(exc)
                metrics.inc("structured_retry", agent=self.name)
                self.log.warning(
                    f"{step}: could not parse the answer (attempt {attempt}/{attempts}): {last_error}",
                    step=step,
                    session_id=state["session_id"],
                    iteration=state.get("iteration", 0),
                    attempt=attempt,
                )
        return None, last_error
