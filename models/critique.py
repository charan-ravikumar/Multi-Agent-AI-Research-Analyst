from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator


class Critique(BaseModel):
    """
    Structured quality assessment produced by the Critic agent.
    overall_score is computed automatically as a weighted average if not supplied.
    If overall_score < settings.critic_score_threshold and iteration < max, the
    orchestrator routes back to the Searcher for another pass.
    """

    factuality_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    citation_quality_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback: str                               # plain-text notes for re-research
    gaps: List[str] = Field(default_factory=list)  # specific topics to re-search
    needs_more_research: bool = True

    @model_validator(mode="after")
    def compute_overall(self) -> "Critique":
        if self.overall_score == 0.0:
            self.overall_score = round(
                0.5 * self.factuality_score
                + 0.3 * self.completeness_score
                + 0.2 * self.citation_quality_score,
                4,
            )
        return self
