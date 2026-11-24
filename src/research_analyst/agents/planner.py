from __future__ import annotations

from typing import Any


from research_analyst.agents.base import BaseAgent, StateLike
from research_analyst.config import settings
from research_analyst.llm.json_parse import parse_json
from research_analyst.safety.sanitize import validate_query
from research_analyst.llm import ModelTier
from research_analyst.models import ResearchPlan
from research_analyst.agents.prompts import planner as prompts

# ── prompts ───────────────────────────────────────────────────────────────────


class PlannerError(RuntimeError):
    """Raised when the LLM fails to return a valid ResearchPlan after retries."""


class PlannerAgent(BaseAgent):
    """
    Decomposes state["query"] into sub-questions and writes:
      state["research_plan"]  — full ResearchPlan Pydantic model
      state["sub_questions"]  — list[str] for the Searcher to fan out over
    """

    name = "planner"
    tier = ModelTier.STRONG
    json_mode = True

    async def run(self, state: StateLike) -> dict[str, Any]:
        query: str = validate_query(state["query"])
        session_id: str = state["session_id"]
        iteration: int = state.get("iteration", 0)

        system = prompts.SYSTEM.format(
            min_q=settings.planner_min_sub_questions,
            max_q=settings.planner_max_sub_questions,
        )

        # conversation history — extended on retry
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompts.USER.format(query=query)},
        ]

        plan, last_error = await self.generate_parsed(
            messages, lambda raw: _parse_and_validate(raw, query), state,
            retry_message=prompts.RETRY, step="plan_validate",
        )

        if plan is None:
            raise PlannerError(
                f"PlannerAgent failed to produce a valid ResearchPlan after 2 attempts. "
                f"Last error: {last_error}"
            )

        self.log.info(
            "plan ready",
            step="plan_complete",
            session_id=session_id,
            iteration=iteration,
            sub_question_count=len(plan.sub_questions),
            sub_questions=plan.sub_questions,
            depth=plan.depth,
        )

        return {
            **state,
            "research_plan": plan,
            "sub_questions": plan.sub_questions,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_and_validate(raw: str, query: str) -> ResearchPlan:
    """
    Strip markdown fences, parse JSON, inject query if missing, validate.
    Raises json.JSONDecodeError, pydantic.ValidationError, or ValueError.
    """
    data = parse_json(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    # allow the model to omit "query" — backfill from state
    data.setdefault("query", query)

    return ResearchPlan.model_validate(data)
