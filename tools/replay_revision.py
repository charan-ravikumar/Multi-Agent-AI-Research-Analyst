"""
tools/replay_revision.py — Replay the revision step against saved state.

Loads logs/last_state.json, calls the CURRENT _REVISION_SYSTEM prompt with
the round-1 draft + round-1 critic objections, and writes the result to
logs/replayed_state.json.  This lets us test just the revision component
without running a full graph (and burning the full TPM budget).

The replayed state is identical to last_state.json except:
  - revised_draft: replaced with the new revision output
  - replay_note: explains this is a single-round replay, not a 3-round debate

Usage:
    python tools/replay_revision.py
    python tools/eval_harness.py --state-file logs/replayed_state.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import the CURRENT revision prompt (after the fix)
from agents.synthesizer import _REVISION_SYSTEM, _REVISION_USER, _parse_revision  # type: ignore[attr-defined]
import llm
from llm import ModelTier

STATE_IN  = ROOT / "logs" / "last_state.json"
STATE_OUT = ROOT / "logs" / "replayed_state.json"


def run_revision(draft_sections: list[str], objections: list[str]) -> list[str] | None:
    """Call the revision LLM with the current (fixed) prompt."""
    draft_text = "\n\n".join(draft_sections) if draft_sections else "(no draft)"
    objections_block = "\n".join(f"{i}. {o}" for i, o in enumerate(objections, 1))

    messages = [
        {"role": "system", "content": _REVISION_SYSTEM},
        {
            "role": "user",
            "content": _REVISION_USER.format(
                n_sections=len(draft_sections),
                draft_text=draft_text,
                n_obj=len(objections),
                objections_block=objections_block,
            ),
        },
    ]

    print("Calling revision LLM (fixed prompt, FAST tier — STRONG tier daily budget exhausted)...")
    raw, usage = llm.generate(messages, tier=ModelTier.FAST, return_usage=True)
    print(f"  tokens: {usage}")
    print(f"  response ({len(raw)} chars):")
    print(f"  {raw[:300]}")

    try:
        sections = _parse_revision(raw)
        print(f"  parsed: {len(sections)} section(s)")
        return sections
    except Exception as e:
        print(f"  PARSE FAILED: {e}")
        print(f"  raw: {repr(raw[:500])}")
        return None


def main() -> None:
    print(f"Loading {STATE_IN}...")
    state = json.loads(STATE_IN.read_text(encoding="utf-8"))

    draft = state.get("draft", [])
    objections = state.get("critic_objections") or []

    print(f"  draft sections: {len(draft)}")
    print(f"  objections: {len(objections)}")
    for i, o in enumerate(objections, 1):
        print(f"    {i}. {o[:90]}")
    print()

    new_revised = run_revision(draft, objections)
    if new_revised is None:
        print("Revision failed — state NOT saved.")
        sys.exit(1)

    print("\nNew revised sections:")
    for i, s in enumerate(new_revised):
        print(f"\n  [{i}] ({len(s)} chars):")
        print(f"  {s[:300]}")

    # Write the replayed state
    replayed = dict(state)
    replayed["revised_draft"] = new_revised
    replayed["replay_note"] = (
        "revised_draft replaced by tools/replay_revision.py using the fixed "
        "_REVISION_SYSTEM prompt — single round replay against round-1 objections, "
        "not a 3-round debate simulation"
    )

    STATE_OUT.write_text(json.dumps(replayed, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved → {STATE_OUT} ({STATE_OUT.stat().st_size // 1024} KB)")
    print("\nNext step:")
    print("  python tools/eval_harness.py --state-file logs/replayed_state.json")


if __name__ == "__main__":
    main()
