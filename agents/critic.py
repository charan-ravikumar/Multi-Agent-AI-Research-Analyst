from __future__ import annotations

import json
import re
from typing import List

from agents.base import BaseAgent
from config import settings
from llm import ModelTier
from models import ResearchState

# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a critical reviewer of assembled research draft reports.

You will receive:
  - The full set of sub-questions the report must address
  - The assembled draft (one or more synthesis paragraphs, one per sub-question)
  - Contradictions already flagged between sources
  - Unresolved knowledge gaps

Your task: identify SPECIFIC, ADDRESSABLE issues with the draft.
An issue is specific and addressable if a revision agent can directly fix it.

Good objections (be this specific):
  - "Claim about X in section 2 relies on a single source; corroboration needed"
  - "Sub-question N ('...') is not answered anywhere in the draft"
  - "Contradiction between sections 1 and 3 ('A vs B') is present but not disclosed"
  - "Section 2 asserts Y without citing any source"

Bad objections (too vague — do NOT produce these):
  - "The draft could be better"
  - "More research needed"
  - "Some claims are unsupported"

Respond with ONLY a valid JSON object — no markdown fences, no prose:

{
  "objections": ["<specific issue 1>", "<specific issue 2>", ...]
}

Rules:
  - If the draft is satisfactory, return: {"objections": []}
  - Limit to at most 10 objections (the most important ones only)
  - Each objection must be at least 15 characters
  - Output nothing except the JSON object.\
"""

_USER = """\
Sub-questions this report must address ({n_sq}):
{sub_questions_block}

Assembled draft ({n_sections} section(s)):
{draft_text}

Contradictions already flagged between sources ({n_contra}):
{contra_block}

Unresolved knowledge gaps ({n_gaps}):
{gaps_block}

Debate round: {debate_round} of {max_rounds}

Review the draft and list any specific, addressable issues.\
"""

_RETRY = (
    "Your previous response could not be parsed or validated.\n"
    "Error: {error}\n\n"
    "Return ONLY the corrected JSON object with no other text."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_MAX_OBJECTIONS = 10
_MIN_OBJECTION_LEN = 15


class CriticError(RuntimeError):
    """Raised when CriticAgent cannot produce a valid response after retries."""


class CriticAgent(BaseAgent):
    """
    Reviews the fully assembled draft (post fan-in) and produces a list of
    specific, addressable objections.  An empty list signals approval.

    Runs inside _debate_loop_node (not as a direct graph node), so the
    debate loop can call it multiple times in a single graph step.

    Reads  : state["draft"]             (List[str] — all branch sections)
             state["contradictions"]    (List[str])
             state["unresolved_gaps"]   (List[str])
             state["sub_questions"]     (List[str])
             state["debate_round"]      (int)
    Writes : state["critic_objections"] (List[str])
    """

    name = "critic"
    tier = ModelTier.STRONG

    async def run(self, state: ResearchState) -> dict:
        session_id: str = state["session_id"]
        iteration: int = state.get("iteration", 0)
        debate_round: int = state.get("debate_round", 0)

        draft_sections: List[str] = list(state.get("draft", []))
        contradictions: List[str] = list(state.get("contradictions", []))
        unresolved_gaps: List[str] = list(state.get("unresolved_gaps", []))
        sub_questions: List[str] = list(state.get("sub_questions", []))

        draft_text = "\n\n".join(draft_sections) if draft_sections else "(no draft produced)"
        sub_qs_block = (
            "\n".join(f"{i}. {sq}" for i, sq in enumerate(sub_questions, 1))
            if sub_questions else "(none)"
        )
        contra_block = (
            "\n".join(f"- {c}" for c in contradictions) if contradictions else "(none)"
        )
        gaps_block = (
            "\n".join(f"- {g}" for g in unresolved_gaps) if unresolved_gaps else "(none)"
        )

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    n_sq=len(sub_questions),
                    sub_questions_block=sub_qs_block,
                    n_sections=len(draft_sections),
                    draft_text=draft_text,
                    n_contra=len(contradictions),
                    contra_block=contra_block,
                    n_gaps=len(unresolved_gaps),
                    gaps_block=gaps_block,
                    debate_round=debate_round + 1,
                    max_rounds=settings.debate_max_rounds,
                ),
            },
        ]

        objections: List[str] | None = None
        last_raw = ""
        last_error = ""

        for attempt in range(1, 3):   # attempt 1 = initial, attempt 2 = retry
            if attempt == 2:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append({"role": "user", "content": _RETRY.format(error=last_error)})

            last_raw, _usage = await self.llm_generate(messages, state=state)

            try:
                objections = _parse_objections(last_raw)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                self.log.warning(
                    "critic validation failed",
                    step="critic_validate",
                    session_id=session_id,
                    iteration=iteration,
                    attempt=attempt,
                    error=last_error,
                )

        if objections is None:
            # Both attempts failed — fail-safe: treat as no objections so the
            # pipeline is never blocked by a persistently unparseable LLM response.
            self.log.error(
                "critic failed to parse after 2 attempts; treating as no objections",
                step="critic_error",
                session_id=session_id,
                iteration=iteration,
            )
            objections = []

        self.log.info(
            f"critic produced {len(objections)} objection(s)",
            step="critic_done",
            session_id=session_id,
            iteration=iteration,
        )

        return {"critic_objections": objections}


# ── module-level helpers ──────────────────────────────────────────────────────

def _parse_objections(raw: str) -> List[str]:
    """Parse and validate {"objections": [...]} from LLM response."""
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)

    data = json.loads(raw.strip())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    if "objections" not in data:
        raise ValueError("Missing 'objections' key in response")

    raw_list = data["objections"]
    if not isinstance(raw_list, list):
        raise ValueError(f"'objections' must be a list, got {type(raw_list).__name__}")

    result: List[str] = []
    for item in raw_list:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if len(item) >= _MIN_OBJECTION_LEN:
            result.append(item)
        if len(result) >= _MAX_OBJECTIONS:
            break

    return result

