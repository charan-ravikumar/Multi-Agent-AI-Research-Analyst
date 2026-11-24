"""GroundingJudgeAgent — LLM check of every cited claim of the FINAL report against its cited evidence."""
from __future__ import annotations

import functools
from typing import Any

from research_analyst.agents.base import BaseAgent, StateLike
from research_analyst.config import settings
from research_analyst.grounding import judge
from research_analyst.grounding.verify import build_evidence_index, extract_claims
from research_analyst.llm import ModelTier
from research_analyst.observability.usage import BudgetExceeded


class GroundingJudgeAgent(BaseAgent):
    """
    Reads  : final_report, references, search_results, extracted_facts
    Writes : claim_verdicts = {"verdicts": {claim_id(str): {verdict, supporting_refs, reason}},
                               "summary": {...counts...}, "judge": {model/status info}}

    Claims are judged in batches (``grounding_judge_batch_size``) with the independent ``judge_model`` (a different
    model family from the generator, same as the evaluation harness).  Failure is never fatal: claims the judge could
    not rate simply have no verdict and the UI falls back to the lexical label, disclosed as such.
    """

    name = "grounding_judge"
    tier = ModelTier.STRONG
    json_mode = True
    max_tokens = 1200

    def __init__(self) -> None:
        super().__init__()
        self.groq_model = settings.judge_model or None

    async def run(self, state: StateLike) -> dict[str, Any]:
        report = state.get("final_report", "")
        refs = state.get("references", [])
        claims = extract_claims(report)[: settings.grounding_judge_max_claims]
        info: dict[str, Any] = {"model": settings.judge_model or "default", "status": "ok", "error": ""}
        if not claims or not refs:
            return {"claim_verdicts": {"verdicts": {}, "summary": judge.summarise({}, len(claims)),
                                       "judge": {**info, "status": "skipped"}}}
        evidence = build_evidence_index(state.get("search_results", []), state.get("extracted_facts", []), refs)
        verdicts: dict[int, dict] = {}
        size = max(1, settings.grounding_judge_batch_size)
        for i in range(0, len(claims), size):
            batch = claims[i:i + size]
            ids = [c.id for c in batch]
            messages = [{"role": "system", "content": judge.SYSTEM},
                        {"role": "user", "content": judge.build_batch_prompt(batch, evidence)}]
            parse = functools.partial(judge.parse_verdicts, expected_ids=set(ids), valid_refs={c.id: c.refs for c in batch})
            try:
                got, err = await self.generate_parsed(
                    messages, parse, state,
                    retry_message=judge.RETRY.replace("{ids}", str(ids)), step="judge_parse")
            except BudgetExceeded:
                info.update(status="partial", error="token budget exhausted")
                break
            except Exception as exc:  # noqa: BLE001 — the judge is an add-on; a provider failure must not fail the run
                info.update(status="partial", error=f"{type(exc).__name__}: {str(exc)[:120]}")
                break
            if got is None:
                info.update(status="partial", error=f"unparseable judge answer: {err[:100]}")
                continue
            verdicts.update(got)
        if not verdicts:
            info["status"] = "failed"
        return {"claim_verdicts": {
            "verdicts": {str(k): v for k, v in verdicts.items()},
            "summary": judge.summarise(verdicts, len(extract_claims(report))),
            "judge": info,
        }}
