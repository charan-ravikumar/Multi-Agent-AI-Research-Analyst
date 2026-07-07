from __future__ import annotations

import json
import re

from pydantic import ValidationError

from agents.base import BaseAgent
from config import settings
from llm import ModelTier
from models import ResearchPlan, ResearchState

# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a research planning assistant.
Your only job is to decompose a research query into focused, independently
researchable sub-questions.

Respond with ONLY a valid JSON object — no markdown fences, no prose, no
explanation before or after. The object must match this schema exactly:

{{
  "query": "<the original query, copied verbatim>",
  "sub_questions": [
    "<sub-question 1>",
    ...
  ],
  "depth": "brief" | "standard" | "deep",
  "strategy_notes": "<one sentence describing your decomposition strategy>"
}}

Constraints:
- sub_questions must contain between {min_q} and {max_q} items.
- Each sub-question must be specific, self-contained, and independently answerable.
- Choose depth based on query complexity: factual → brief, analytical → standard,
  interdisciplinary → deep.
- Output nothing except the JSON object.\
"""

_USER = "Research query: {query}"

_RETRY = (
    "Your previous response could not be parsed or validated.\n"
    "Error: {error}\n\n"
    "Return ONLY the corrected JSON object with no other text."
)

# strips ```json ... ``` or ``` ... ``` wrappers that models sometimes emit
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


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

    async def run(self, state: ResearchState) -> ResearchState:
        query: str = state["query"]
        session_id: str = state["session_id"]
        iteration: int = state.get("iteration", 0)

        system = _SYSTEM.format(
            min_q=settings.planner_min_sub_questions,
            max_q=settings.planner_max_sub_questions,
        )

        # conversation history — extended on retry
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": _USER.format(query=query)},
        ]

        plan: ResearchPlan | None = None
        last_raw = ""
        last_error = ""

        for attempt in range(1, 3):   # attempt 1 = initial, attempt 2 = retry
            if attempt == 2:
                # append the bad response and the specific error so the model
                # knows exactly what to fix
                messages.append({"role": "assistant", "content": last_raw})
                messages.append({"role": "user", "content": _RETRY.format(error=last_error)})

            last_raw, _usage = await self.llm_generate(messages, state=state)

            try:
                plan = _parse_and_validate(last_raw, query)
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                self.log.warning(
                    "plan validation failed",
                    step="plan_validate",
                    session_id=session_id,
                    iteration=iteration,
                    attempt=attempt,
                    error=last_error,
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

def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text.strip()


def _parse_and_validate(raw: str, query: str) -> ResearchPlan:
    """
    Strip markdown fences, parse JSON, inject query if missing, validate.
    Raises json.JSONDecodeError, pydantic.ValidationError, or ValueError.
    """
    cleaned = _strip_fences(raw)
    data: dict = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    # allow the model to omit "query" — backfill from state
    data.setdefault("query", query)

    return ResearchPlan.model_validate(data)
