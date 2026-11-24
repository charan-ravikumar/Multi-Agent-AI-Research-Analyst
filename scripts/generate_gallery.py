"""
Generate the bundled sample reports by running the REAL pipeline on the example questions.

    uv run python scripts/generate_gallery.py                 # all example questions
    uv run python scripts/generate_gallery.py --only 1 3      # by index
    uv run python scripts/generate_gallery.py --max-tokens 50000

LIVE: uses LLM/search quota (about 20-30k tokens per report).  Stops if a run fails; existing entries are
overwritten only for the questions generated.  Review the output before committing it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_analyst.config import settings  # noqa: E402
from research_analyst.llm.preflight import assert_models_available  # noqa: E402
from research_analyst.models import ResearchState  # noqa: E402
from research_analyst.observability import tracing  # noqa: E402
from research_analyst.pipeline.graph import app  # noqa: E402
from research_analyst.ui.about import SAMPLE_QUESTIONS  # noqa: E402
from research_analyst.ui.gallery import GALLERY_DIR, entry_from_state  # noqa: E402


def initial_state(question: str) -> ResearchState:
    return ResearchState(
        session_id=str(uuid.uuid4()), query=question, research_plan=None, sub_questions=[], search_results=[],
        failed_sub_questions=[], extracted_facts=[], citations=[], contradictions=[], unresolved_gaps=[], gap_items=[],
        draft=[], final_report="", iteration=0, routing_history=[], auto_approve_plan=True)


async def generate(question: str) -> dict:
    st = initial_state(question)
    t0 = time.time()
    with tracing.run_span("gallery", st["session_id"], question, tags=["gallery"]):
        state = await asyncio.wait_for(app.ainvoke(st, config={"configurable": {"thread_id": st["session_id"]}}),
                                       settings.run_timeout_seconds)
        tracing.record_scores(state)
    return entry_from_state(state, question, time.time() - t0,
                            {"fast": settings.gemini_fast_model, "strong": settings.gemini_strong_model})


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", help="1-based indexes of SAMPLE_QUESTIONS")
    ap.add_argument("--max-tokens", type=int, default=60_000, help="per-run token budget")
    a = ap.parse_args()
    assert_models_available()
    settings.token_budget_per_run = a.max_tokens
    GALLERY_DIR.mkdir(exist_ok=True)
    picks = [SAMPLE_QUESTIONS[i - 1] for i in a.only] if a.only else SAMPLE_QUESTIONS
    for q in picks:
        print(f"→ {q}", flush=True)
        entry = await generate(q)
        s = entry["citation_stats"]
        print(f"   {entry['seconds']}s  {entry['usage']['total_tokens']} tokens  cited {s.get('cited')}/{s.get('claim_sentences')}  "
              f"refs {len(entry['references'])}  coverage {len(entry['coverage'])}  follow-ups {len(entry['open_questions'])}", flush=True)
        (GALLERY_DIR / f"{entry['id']}.json").write_text(json.dumps(entry, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
