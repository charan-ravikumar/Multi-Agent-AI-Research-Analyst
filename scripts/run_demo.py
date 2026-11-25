"""
tools/run_demo.py — Run one real graph invocation and print the session_id.

This makes live Gemini LLM calls with real web search (DuckDuckGo).
After it completes, run:
    python tools/trace_summary.py --last
to see the full decision timeline.

Rate-limit note: SEARCH_MAX_RESULTS is set to 3 and READER_MAX_CONCURRENT_LLM_CALLS
to 1 so the parallel Reader branches stay within the free-tier 15 requests/minute limit.
"""
from __future__ import annotations

from typing import Any

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Set BEFORE any project imports so Pydantic BaseSettings picks them up.
os.environ["SEARCH_MAX_RESULTS"] = "3"
os.environ["READER_MAX_CONCURRENT_LLM_CALLS"] = "1"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))   # allow running without installing the package

from research_analyst.pipeline.graph import app
from research_analyst.models.research_state import new_research_state


def _save_state(path: Path, text: str) -> int:
    path.write_text(text, encoding="utf-8")
    return path.stat().st_size // 1024


async def main() -> None:
    from research_analyst.llm.preflight import assert_models_available
    assert_models_available()           # fail before spending quota if a configured model was retired
    session_id = str(uuid.uuid4())
    thread_id  = str(uuid.uuid4())

    print(f"session_id : {session_id}")
    print(f"thread_id  : {thread_id}")
    print("query      : What are the main applications of large language models in healthcare?")
    print()

    initial = new_research_state(
        session_id, "What are the main applications of large language models in healthcare?",
        auto_approve_plan=True,   # skip human checkpoint for this demo
    )

    cfg = {"configurable": {"thread_id": thread_id}}

    print("Running graph (real LLM + web search)...")
    from research_analyst.config import settings
    from research_analyst.observability import tracing
    with tracing.run_span("demo", session_id, initial["query"]):
        result = await asyncio.wait_for(app.ainvoke(initial, config=cfg), timeout=settings.run_timeout_seconds)
        tracing.record_scores(result)

    # ── Save full state for eval_harness.py ──────────────────────────────────
    import json as _json
    from pathlib import Path as _Path

    state_path = _Path("logs/last_state.json")
    state_path.parent.mkdir(exist_ok=True)

    def _serialise(obj: Any) -> Any:
        """Recursively serialise Pydantic models and other non-JSON types."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, list):
            return [_serialise(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        return obj

    state_json = {k: _serialise(v) for k, v in result.items()}
    saved_kb = await asyncio.to_thread(_save_state, state_path, _json.dumps(state_json, indent=2, default=str))
    print(f"\nState saved → {state_path}  ({saved_kb} KB)")
    # ─────────────────────────────────────────────────────────────────────────

    routing = result.get("routing_history", [])
    print(f"\nDone.  routing_history ({len(routing)} entries):")
    for entry in routing:
        print(f"  {entry}")

    report = result.get("final_report", "")
    print(f"\nfinal_report ({len(report)} chars):")
    print(report[:800])
    if len(report) > 800:
        print(f"  ... [{len(report) - 800} more chars]")

    print(f"\n\nTo inspect this run:\n  python tools/trace_summary.py --session-id {session_id}")
    print("Or simply:\n  python tools/trace_summary.py --last")


if __name__ == "__main__":
    asyncio.run(main())
