from __future__ import annotations

from pydantic import BaseModel, field_validator


class SearchResult(BaseModel):
    """
    A single result returned by the Searcher agent.
    full_text is populated after Playwright scraping; may be empty if
    the page was unreachable or scraping was skipped.
    """

    sub_question: str           # which sub-question this result addresses
    url: str
    title: str
    snippet: str                # search-engine summary blurb
    full_text: str = ""         # scraped body text
    source_type: str = "web"    # "web" | "arxiv" | "semantic_scholar"

    @field_validator("source_type")
    @classmethod
    def valid_source_type(cls, v: str) -> str:
        allowed = {"web", "arxiv", "semantic_scholar"}
        if v not in allowed:
            raise ValueError(f"source_type must be one of {allowed}")
        return v
