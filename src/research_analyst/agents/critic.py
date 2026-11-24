from __future__ import annotations


from research_analyst.agents.base import BaseAgent, StateLike
from research_analyst.observability.prompt_store import resolve
from research_analyst.grounding.citations import analyze, build_evidence
from research_analyst.grounding.gaps import curate_gaps, gap_texts
from research_analyst.llm.json_parse import parse_json
from research_analyst.config import settings
from research_analyst.llm import ModelTier
from research_analyst.agents.prompts import critic as prompts

# ── prompts ───────────────────────────────────────────────────────────────────


_MAX_OBJECTIONS = 6
_SEVERITIES = ("high", "medium", "low")
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
    Writes : critic_objections (high+medium, for the reviser), critic_blocking (# high),
             critic_minor (low-severity notes), critic_review_skipped
    """

    name = "critic"
    tier = ModelTier.STRONG
    json_mode = True

    def __init__(self) -> None:
        super().__init__()
        # A different model family than the (Groq) Synthesizer reduces shared blind spots.
        self.prefer_provider = "gemini" if settings.critic_prefer_gemini else None
        self.groq_model = settings.critic_model or None

    async def run(self, state: StateLike) -> dict:
        session_id: str = state["session_id"]
        iteration: int = state.get("iteration", 0)
        debate_round: int = state.get("debate_round", 0)

        draft_sections: list[str] = list(state.get("draft", []))
        contradictions: list[str] = list(state.get("contradictions", []))
        gap_items = state.get("gap_items") or []
        unresolved_gaps: list[str] = gap_texts(curate_gaps(gap_items)) if gap_items else list(state.get("unresolved_gaps", []))
        sub_questions: list[str] = list(state.get("sub_questions", []))

        prev: list[str] = list(state.get("previous_objections", []))
        checks = analyze(draft_sections, state.get("extracted_facts", []))
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
            {"role": "system", "content": resolve("critic-system").text},
            {
                "role": "user",
                "content": prompts.USER.format(
                    n_sq=len(sub_questions),
                    sub_questions_block=sub_qs_block,
                    evidence_block=build_evidence(
                        state.get("extracted_facts", []), settings.evidence_block_max_chars),
                    n_checks=len(checks["problems"]),
                    cited=checks["cited"],
                    claims=checks["claim_sentences"],
                    checks_block="\n".join(f"- {p}" for p in checks["problems"]) or "(none)",
                    n_sections=len(draft_sections),
                    draft_text=draft_text,
                    n_contra=len(contradictions),
                    contra_block=contra_block,
                    n_gaps=len(unresolved_gaps),
                    gaps_block=gaps_block,
                    n_prev=len(prev),
                    prev_block="\n".join(f"- {p}" for p in prev) or "(none — first review)",
                    debate_round=debate_round + 1,
                    max_rounds=int(state.get("review_rounds", settings.debate_max_rounds)),
                ),
            },
        ]

        parsed, _err = await self.generate_parsed(
            messages, _parse_objections, state, retry_message=prompts.RETRY, step="critic_validate",
        )

        if parsed is None:
            # Both attempts failed — never block the pipeline, but do NOT let the
            # caller believe the draft was approved: flag the review as skipped.
            self.log.error(
                "critic failed to parse after 2 attempts; review SKIPPED (not approved)",
                step="critic_error",
                session_id=session_id,
                iteration=iteration,
            )
            parsed = []
            review_skipped = True
        else:
            review_skipped = False

        # high+medium go to the reviser; only HIGH blocks acceptance; low = disclosed notes.
        objections = [o["text"] for o in parsed if o["severity"] in ("high", "medium")]
        blocking = sum(1 for o in parsed if o["severity"] == "high")
        minor = [o["text"] for o in parsed if o["severity"] == "low"]

        self.log.info(
            f"critic produced {len(objections)} actionable objection(s) ({blocking} high), {len(minor)} minor",
            step="critic_done",
            session_id=session_id,
            iteration=iteration,
        )

        return {
            "critic_objections": objections,
            "critic_blocking": blocking,
            "critic_minor": minor,
            "critic_review_skipped": review_skipped,
        }


# ── module-level helpers ──────────────────────────────────────────────────────

def _parse_objections(raw: str) -> list[dict]:
    """Parse {"objections": [...]} -> [{"severity","text"}].  Plain strings are accepted
    (severity defaults to "high" so older/simpler critics keep blocking behaviour)."""
    data = parse_json(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    if "objections" not in data:
        raise ValueError("Missing 'objections' key in response")

    raw_list = data["objections"]
    if not isinstance(raw_list, list):
        raise ValueError(f"'objections' must be a list, got {type(raw_list).__name__}")

    result: list[dict] = []
    for item in raw_list:
        if isinstance(item, str):
            sev, text = "high", item
        elif isinstance(item, dict):
            sev = str(item.get("severity", "high")).strip().lower()
            text = str(item.get("objection") or item.get("text") or "")
        else:
            continue
        text = text.strip()
        if sev not in _SEVERITIES:
            sev = "high"
        if len(text) >= _MIN_OBJECTION_LEN:
            result.append({"severity": sev, "text": text})
        if len(result) >= _MAX_OBJECTIONS:
            break
    return result
