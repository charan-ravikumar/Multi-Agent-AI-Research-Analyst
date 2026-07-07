from __future__ import annotations

from pydantic import BaseModel, HttpUrl, field_validator


class Citation(BaseModel):
    """A citable source referenced in the final report."""

    url: str
    title: str
    snippet: str                # most relevant excerpt from the source
    source_type: str = "web"    # "web" | "arxiv" | "semantic_scholar"

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("url must not be empty")
        return v.strip()

    @field_validator("source_type")
    @classmethod
    def valid_source_type(cls, v: str) -> str:
        allowed = {"web", "arxiv", "semantic_scholar"}
        if v not in allowed:
            raise ValueError(f"source_type must be one of {allowed}")
        return v
