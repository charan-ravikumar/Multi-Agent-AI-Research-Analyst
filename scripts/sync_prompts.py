"""
Push the in-repo SYSTEM prompts to Langfuse Prompt Management.

    uv run python scripts/sync_prompts.py --dry-run              # show what would change
    uv run python scripts/sync_prompts.py                        # create versions labelled "production"
    uv run python scripts/sync_prompts.py --label staging        # try a prompt edit without touching production

Idempotent: a prompt whose text at that label already matches the repo is left alone.  Requires LANGFUSE_* keys.
Workflow: edit the prompt in the repo -> sync to "staging" -> run the evaluation with LANGFUSE_PROMPT_LABEL=staging
-> if it wins, move the "production" label to that version in the Langfuse UI (or sync with --label production).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from research_analyst.config import settings  # noqa: E402
from research_analyst.observability import prompt_sync, tracing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=settings.langfuse_prompt_label)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-m", "--message", default="sync from repo")
    a = ap.parse_args()
    client = tracing.get_client()
    if client is None:
        print("Langfuse is not configured (set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL).")
        return 2
    results = prompt_sync.sync(client, a.label, dry_run=a.dry_run, commit_message=a.message)
    for r in results:
        print(f"{r.action:13} {r.name:30} {('v' + str(r.version)) if r.version else ''} {r.detail}")
    client.flush()
    return 1 if any(r.action == "error" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
