"""
tests/test_writer_manual.py

Full chain: search_web -> ReaderAgent -> SynthesizerAgent -> WriterAgent.
Prints the complete final_report and explicitly verifies that the 3 gaps
and 0 contradictions from the Synthesizer appear in the disclosure sections.
Run from the project root:

    .venv/Scripts/python.exe tests/test_writer_manual.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.reader import ReaderAgent
from agents.synthesizer import SynthesizerAgent
from agents.writer import WriterAgent
from models import ResearchState
from tools.web_search import search_web


async def main() -> None:
    sub_question = "How does AI accelerate drug target identification?"
    session_id = str(uuid.uuid4())

    print(f"\n{'='*65}")
    print(f"Sub-question : {sub_question}")
    print(f"Session      : {session_id[:8]}...")
    print(f"{'='*65}\n")

    # ── Step 1: search ────────────────────────────────────────────────────────
    print("Step 1: Fetching search results...")
    search_results = await search_web(sub_question, sub_question=sub_question, max_results=3)
    print(f"  Got {len(search_results)} result(s).\n")

    state: ResearchState = ResearchState(
        session_id=session_id,
        query="What is the impact of AI on drug discovery timelines?",
        research_plan=None,
        sub_questions=[sub_question],
        current_sub_question=sub_question,
        search_results=search_results,
        failed_sub_questions=[],
        extracted_facts=[],
        citations=[],
        contradictions=[],
        unresolved_gaps=[],
        draft=[],
        critique=None,
        final_report="",
        iteration=0,
    )

    # ── Step 2: extract facts ─────────────────────────────────────────────────
    print("Step 2: ReaderAgent...")
    state = await ReaderAgent()(state)
    print(f"  Extracted {len(state['extracted_facts'])} fact(s).\n")

    # ── Step 3: synthesize ────────────────────────────────────────────────────
    print("Step 3: SynthesizerAgent...")
    state = await SynthesizerAgent()(state)
    print(f"  Draft sections:  {len(state['draft'])}")
    print(f"  Contradictions:  {state['contradictions']}")
    print(f"  Unresolved gaps: {len(state['unresolved_gaps'])}")
    for g in state['unresolved_gaps']:
        print(f"    • {g}")
    print()

    # ── Step 4: write ─────────────────────────────────────────────────────────
    print("Step 4: WriterAgent...\n")
    state = await WriterAgent()(state)

    report = state["final_report"]

    print(f"{'═'*65}")
    print("FINAL REPORT (full text):")
    print(f"{'═'*65}\n")
    print(report)
    print(f"\n{'═'*65}\n")

    # ── Section 5.9 disclosure verification ──────────────────────────────────
    print("Section 5.9 disclosure verification:")
    print(f"{'─'*65}")

    report_lower = report.lower()

    # contradictions
    n_contra = len(state["contradictions"])
    contra_header_present = "contradictions" in report_lower
    print(f"  contradictions in state   : {n_contra}")
    print(f"  'contradictions' header   : {'YES' if contra_header_present else 'NO — MISSING'}")
    if n_contra == 0:
        no_contra_text = "no contradictions were identified" in report_lower
        print(f"  'no contradictions' text  : {'YES' if no_contra_text else 'NO — MISSING'}")

    print()

    # gaps
    n_gaps = len(state["unresolved_gaps"])
    gaps_header_present = "knowledge gaps" in report_lower
    print(f"  unresolved_gaps in state  : {n_gaps}")
    print(f"  'knowledge gaps' header   : {'YES' if gaps_header_present else 'NO — MISSING'}")
    if n_gaps > 0:
        for i, gap in enumerate(state["unresolved_gaps"], 1):
            # check for a meaningful substring (first 30 chars) of each gap
            key = gap[:40].lower()
            present = key in report_lower
            print(f"  gap {i} present in report : {'YES' if present else 'NO — NOT FOUND'} | {gap[:60]}...")

    print(f"{'─'*65}\n")

    # ── assertions ────────────────────────────────────────────────────────────
    assert isinstance(report, str) and report.strip(), \
        "final_report must be a non-empty string"

    for header in ["executive summary", "key findings", "detailed analysis",
                   "contradictions", "knowledge gaps", "references"]:
        assert header in report_lower, f"Report missing required section: '{header}'"

    if n_contra == 0:
        assert "no contradictions" in report_lower, \
            "Report must explicitly state no contradictions were found"

    assert gaps_header_present, "Report missing 'Knowledge Gaps' section"

    print("All assertions passed.")
    print(f"Final report: {len(report)} characters, "
          f"{len(report.splitlines())} lines.")


if __name__ == "__main__":
    asyncio.run(main())
