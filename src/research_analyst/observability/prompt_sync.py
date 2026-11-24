"""Push the in-repo prompts to Langfuse as new versions (idempotent).  CLI: scripts/sync_prompts.py."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from research_analyst.observability import prompt_store

# Importing these modules registers their prompts (registration happens at import time, next to the text).
_MODULES = (
    "research_analyst.agents.prompts.planner", "research_analyst.agents.prompts.reader",
    "research_analyst.agents.prompts.synthesizer", "research_analyst.agents.prompts.critic",
    "research_analyst.agents.prompts.writer", "research_analyst.retrieval.query",
    "research_analyst.agents.react_search", "research_analyst.grounding.judge",
)


@dataclass
class SyncResult:
    name: str
    action: str          # "created" | "unchanged" | "would-create" | "error"
    version: int | None = None
    detail: str = ""


def load_all() -> dict[str, str]:
    for m in _MODULES:
        importlib.import_module(m)
    return prompt_store.registered()


def sync(client: Any, label: str, *, dry_run: bool = False, commit_message: str = "sync from repo") -> list[SyncResult]:
    """For each registered prompt: create a new Langfuse version carrying ``label`` unless the text at that label is identical."""
    out: list[SyncResult] = []
    for name, text in sorted(load_all().items()):
        try:
            existing = None
            try:
                got = client.get_prompt(name, label=label, type="text", cache_ttl_seconds=0, max_retries=0, fetch_timeout_seconds=5)
                existing = got if not getattr(got, "is_fallback", False) else None
            except Exception:  # noqa: BLE001 — not found is the normal first-sync case
                existing = None
            if existing is not None and existing.prompt == text:
                out.append(SyncResult(name, "unchanged", existing.version))
            elif dry_run:
                out.append(SyncResult(name, "would-create", None, "new" if existing is None else f"differs from v{existing.version}"))
            else:
                created = client.create_prompt(name=name, prompt=text, labels=[label], type="text", commit_message=commit_message)
                out.append(SyncResult(name, "created", getattr(created, "version", None)))
        except Exception as exc:  # noqa: BLE001
            out.append(SyncResult(name, "error", None, f"{type(exc).__name__}: {str(exc)[:120]}"))
    return out
