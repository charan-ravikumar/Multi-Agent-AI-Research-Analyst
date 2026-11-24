"""
observability/prompt_store.py — Langfuse Prompt Management with a local fallback.

Every system prompt is registered here under a stable name (``register``).  At run time an agent asks for it with
``resolve(name)``:

    Langfuse enabled + prompt found under LANGFUSE_PROMPT_LABEL   -> the Langfuse text, its version, and a handle
                                                                     that LINKS each generation to that exact version
    otherwise (disabled, not found, offline, invalid)             -> the in-repo text (the same string as before)

So the repo is always a working fallback and Langfuse is an override: edit a prompt in the Langfuse UI, move the
``production`` label to roll forward or back, point an evaluation at ``staging`` — no deploy.

Safety rails (a prompt edited in a web UI must not be able to crash or silently break a run):
  * prompts that are ``str.format`` templates (``formatted=True``) are only accepted when the remote text has EXACTLY
    the placeholders of the local one; otherwise the local text is used and a warning is logged once;
  * a failing/slow Langfuse is skipped for ``_DOWN_SECONDS`` after one failure (no per-call timeouts), and the SDK's
    own cache (LANGFUSE_PROMPT_CACHE_SECONDS) means a normal call is an in-memory lookup;
  * nothing here can raise into a run: any error returns the local prompt.

Only SYSTEM prompts are managed.  The USER templates stay in code because they are tied to the placeholders and
data the code passes in.
"""
from __future__ import annotations

import logging
import string
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from research_analyst.config import settings
from research_analyst.observability import tracing

_log = logging.getLogger(__name__)

_DOWN_SECONDS = 60.0
_FETCH_TIMEOUT_S = 3
_MAX_TEXT_MEMO = 128

_local: dict[str, str] = {}
_formatted: dict[str, frozenset[str]] = {}
_down_until = 0.0
_warned: set[str] = set()
_by_text: OrderedDict[str, Resolved] = OrderedDict()
_used: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class Resolved:
    name: str
    text: str
    source: str                        # "langfuse" | "local"
    version: int | None = None
    label: str | None = None
    lf: Any = field(default=None, repr=False, compare=False)    # Langfuse prompt client (links generations to the version)

    def format(self, **kw: Any) -> str:
        """``str.format`` the template and remember the result so the generation can still be linked to this prompt."""
        out = self.text.format(**kw)
        _remember(out, self)
        return out


def _fields(template: str) -> frozenset[str]:
    return frozenset(f for _, f, _, _ in string.Formatter().parse(template) if f)


def register(name: str, text: str, *, formatted: bool = False) -> str:
    """Declare an in-repo prompt (the fallback and the source ``scripts/sync_prompts.py`` pushes).  Returns ``text``."""
    _local[name] = text
    if formatted:
        _formatted[name] = _fields(text)
    return text


def registered() -> dict[str, str]:
    return dict(_local)


def _remember(text: str, res: Resolved) -> None:
    _by_text[text] = res
    _by_text.move_to_end(text)
    while len(_by_text) > _MAX_TEXT_MEMO:
        _by_text.popitem(last=False)


def lookup(text: str) -> Resolved | None:
    """The managed prompt a system message came from (None for unmanaged text) — used by BaseAgent for trace linking."""
    return _by_text.get(text)


def used() -> dict[str, dict[str, Any]]:
    """{name: {source, version, label}} for every prompt resolved since the last reset (recorded in eval results)."""
    return {k: dict(v) for k, v in _used.items()}


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        _log.warning(msg)


def _local_prompt(name: str) -> Resolved:
    res = Resolved(name=name, text=_local[name], source="local")
    _remember(res.text, res)
    _used[name] = {"source": "local", "version": None, "label": None}
    return res


def resolve(name: str) -> Resolved:
    """Return the prompt to use right now.  Never raises (except KeyError for a name nobody registered)."""
    global _down_until
    if name not in _local:
        raise KeyError(f"prompt {name!r} is not registered (prompt_store.register)")
    if not settings.langfuse_prompts or time.monotonic() < _down_until:
        return _local_prompt(name)
    client = tracing.get_client()
    if client is None:
        return _local_prompt(name)
    try:
        p = client.get_prompt(
            name, label=settings.langfuse_prompt_label, type="text", fallback=_local[name],
            cache_ttl_seconds=settings.langfuse_prompt_cache_seconds, max_retries=1, fetch_timeout_seconds=_FETCH_TIMEOUT_S)
        text = getattr(p, "prompt", None)
        if getattr(p, "is_fallback", False):
            _down_until = time.monotonic() + _DOWN_SECONDS
            _warn_once(f"missing:{name}", f"Langfuse prompt {name!r} (label {settings.langfuse_prompt_label!r}) unavailable; "
                                          "using the in-repo prompt (push it with scripts/sync_prompts.py)")
            return _local_prompt(name)
        if not isinstance(text, str) or not text.strip():
            _warn_once(f"empty:{name}", f"Langfuse prompt {name!r} v{getattr(p, 'version', '?')} is empty; using the in-repo prompt")
            return _local_prompt(name)
        if name in _formatted and _fields(text) != _formatted[name]:
            _warn_once(f"fields:{name}:{getattr(p, 'version', '?')}",
                       f"Langfuse prompt {name!r} v{getattr(p, 'version', '?')} has placeholders {sorted(_fields(text))} "
                       f"but the code fills {sorted(_formatted[name])}; using the in-repo prompt")
            return _local_prompt(name)
    except Exception as exc:  # noqa: BLE001 — a prompt store problem must never break a run
        _down_until = time.monotonic() + _DOWN_SECONDS
        _warn_once(f"err:{name}", f"Langfuse prompt fetch failed ({type(exc).__name__}: {str(exc)[:100]}); using in-repo prompts for {_DOWN_SECONDS:.0f}s")
        return _local_prompt(name)
    res = Resolved(name=name, text=text, source="langfuse", version=getattr(p, "version", None),
                   label=settings.langfuse_prompt_label, lf=p)
    _remember(text, res)
    _used[name] = {"source": "langfuse", "version": res.version, "label": res.label}
    return res


def reset_for_tests() -> None:
    global _down_until
    _down_until = 0.0
    _warned.clear()
    _by_text.clear()
    _used.clear()
