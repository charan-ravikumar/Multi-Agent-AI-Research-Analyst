from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from config import settings


class ResearchPlan(BaseModel):
    """
    Output of the Planner agent.
    Drives how many sub-questions are generated and the research depth.
    """

    query: str
    sub_questions: List[str] = Field(
        min_length=settings.planner_min_sub_questions,
        max_length=settings.planner_max_sub_questions,
    )
    depth: Literal["brief", "standard", "deep"] = "standard"
    strategy_notes: str = ""    # Planner's reasoning, visible in the Streamlit UI
