from __future__ import annotations

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """
    A single structured fact extracted by the Reader agent from a SearchResult.
    Stored directly in ResearchState and mirrored to Redis.
    """

    sub_question: str                           # which sub-question this answers
    content: str                                # the extracted claim / statement
    source_url: str
    source_title: str
    confidence: float = Field(ge=0.0, le=1.0)  # Reader's self-assessed confidence
