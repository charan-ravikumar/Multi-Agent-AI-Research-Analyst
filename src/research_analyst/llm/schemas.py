"""
llm/schemas.py — response schemas for structured outputs.

Passed to the Gemini API as ``response_schema`` so the model is constrained to this exact shape (valid JSON, required
fields, enums).  Agents still validate and apply their own business rules afterwards (e.g. a verdict id must belong to
the batch): the API guarantees SHAPE, not meaning.

Kept deliberately plain (no validators, no length constraints) because the provider supports only a subset of JSON
Schema; ranges such as "3-5 sub-questions" are enforced by the agent's own model validation, which also retries.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Depth(str, Enum):
    brief = "brief"
    standard = "standard"
    deep = "deep"


class PlannerOut(BaseModel):
    query: str
    sub_questions: list[str]
    depth: Depth
    strategy_notes: str


class ReaderFact(BaseModel):
    source: int
    content: str
    confidence: float


class ReaderOut(BaseModel):
    facts: list[ReaderFact]


class SynthOut(BaseModel):
    synthesis: str
    contradictions: list[str]
    gaps: list[str]


class ReviseOut(BaseModel):
    revised_sections: list[str]


class Severity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Objection(BaseModel):
    severity: Severity
    objection: str


class CriticOut(BaseModel):
    objections: list[Objection]


class RewriteOut(BaseModel):
    queries: list[str]


class Verdict(str, Enum):
    supported = "supported"
    partial = "partial"
    unsupported = "unsupported"


class VerdictItem(BaseModel):
    id: int
    verdict: Verdict
    supporting_refs: list[int]
    reason: str


class JudgeOut(BaseModel):
    verdicts: list[VerdictItem]


class EvalClaim(BaseModel):
    id: int
    verdict: Verdict


class EvalJudgeOut(BaseModel):
    claims: list[EvalClaim]
    completeness: int
    clarity: int
