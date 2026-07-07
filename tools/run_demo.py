"""
tools/run_demo.py — Run one real graph invocation and print the session_id.

This makes a live Groq LLM call with real web search (DuckDuckGo).
After it completes, run:
    python tools/trace_summary.py --last
to see the full decision timeline.

Rate-limit note: SEARCH_MAX_RESULTS is set to 3 and READER_MAX_CONCURRENT_LLM_CALLS
to 1 so the parallel Reader branches stay within Groq's free-tier 6,000 TPM limit.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Set BEFORE any project imports so Pydantic BaseSettings picks them up.
os.environ["SEARCH_MAX_RESULTS"] = "3"
os.environ["READER_MAX_CONCURRENT_LLM_CALLS"] = "1"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.graph import app
from models import ResearchState


async def main() -> None:
    session_id = str(uuid.uuid4())
    thread_id  = str(uuid.uuid4())

    print(f"session_id : {session_id}")
    print(f"thread_id  : {thread_id}")
    print(f"query      : What are the main applications of large language models in healthcare?")
    print()

    initial = ResearchState(
        session_id=session_id,
        query="What are the main applications of large language models in healthcare?",
        research_plan=None,
        sub_questions=[],
        search_results=[],
        failed_sub_questions=[],
        extracted_facts=[],
        citations=[],
        contradictions=[],
        unresolved_gaps=[],
        draft=[],
        critique=None,
        final_report="",
        iteration=0,
        routing_history=[],
        auto_approve_plan=True,   # skip human checkpoint for this demo
    )

    cfg = {"configurable": {"thread_id": thread_id}}

    print("Running graph (real LLM + web search)...")
    result = await app.ainvoke(initial, config=cfg)

    # ── Save full state for eval_harness.py ──────────────────────────────────
    import json as _json
    from pathlib import Path as _Path

    state_path = _Path("logs/last_state.json")
    state_path.parent.mkdir(exist_ok=True)

    def _serialise(obj):
        """Recursively serialise Pydantic models and other non-JSON types."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, list):
            return [_serialise(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        return obj

    state_json = {k: _serialise(v) for k, v in result.items()}
    state_path.write_text(_json.dumps(state_json, indent=2, default=str), encoding="utf-8")
    print(f"\nState saved → {state_path}  ({state_path.stat().st_size // 1024} KB)")
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
    print(f"Or simply:\n  python tools/trace_summary.py --last")


if __name__ == "__main__":
    asyncio.run(main())
