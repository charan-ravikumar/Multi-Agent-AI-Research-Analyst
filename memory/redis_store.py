"""
memory/redis_store.py — Redis-only memory layer

Provides helpers for:
  - session state  (agent scratchpads, short-term working memory)
  - extracted facts (structured records persisted per research session)

Keys follow the pattern:  <prefix>:<session_id>:<field>
All values are JSON-serialised so callers work with plain Python dicts/lists.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis

from config import settings

logger = logging.getLogger(__name__)

# ── singleton client ──────────────────────────────────────────────────────────

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    """Return a lazily-initialised Redis client (connection pooled)."""
    global _client
    if _client is None:
        _client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        _client.ping()          # fail fast if Redis is unreachable
        logger.debug("Redis connected: %s", settings.redis_url)
    return _client


# ── low-level helpers ─────────────────────────────────────────────────────────

def _key(*parts: str) -> str:
    return ":".join(parts)


def set_value(session_id: str, field: str, value: Any, ttl: int | None = None) -> None:
    """Store a JSON-serialisable value under session_id:field."""
    r = get_client()
    k = _key(session_id, field)
    r.set(k, json.dumps(value), ex=ttl or settings.cache_ttl_seconds)


def get_value(session_id: str, field: str) -> Any | None:
    """Retrieve and deserialise a value, or None if missing/expired."""
    r = get_client()
    raw = r.get(_key(session_id, field))
    if raw is None:
        return None
    return json.loads(raw)


def delete_session(session_id: str) -> None:
    """Remove all keys belonging to a session."""
    r = get_client()
    pattern = _key(session_id, "*")
    keys = r.keys(pattern)
    if keys:
        r.delete(*keys)
    logger.debug("Deleted %d keys for session %s", len(keys), session_id)


# ── agent scratchpad ──────────────────────────────────────────────────────────

def save_scratchpad(session_id: str, agent_name: str, data: dict) -> None:
    """Persist an agent's working notes for the current session."""
    set_value(session_id, f"scratchpad:{agent_name}", data)


def load_scratchpad(session_id: str, agent_name: str) -> dict:
    """Load an agent's scratchpad, returning an empty dict if absent."""
    return get_value(session_id, f"scratchpad:{agent_name}") or {}


# ── extracted facts ───────────────────────────────────────────────────────────

def append_facts(session_id: str, facts: list[dict]) -> None:
    """
    Append structured fact records to this session's fact list.
    Each fact dict should include at minimum: source, content, sub_question.
    """
    existing: list[dict] = get_value(session_id, "extracted_facts") or []
    existing.extend(facts)
    set_value(session_id, "extracted_facts", existing)


def get_facts(session_id: str) -> list[dict]:
    """Return all extracted facts accumulated so far for this session."""
    return get_value(session_id, "extracted_facts") or []


def clear_facts(session_id: str) -> None:
    """Reset the fact list (e.g. when starting a fresh research pass)."""
    set_value(session_id, "extracted_facts", [])


# ── cross-run report memory ───────────────────────────────────────────────────
#
# Stores a compact record for every completed report so that later runs can
# surface topically related work from previous sessions.
#
# Key structure:
#   reports:index               — sorted set; score = Unix timestamp;
#                                 member = report_id.  Sorted sets allow
#                                 O(log N) ZADD and O(log N + K) ZREVRANGE,
#                                 so we retrieve the K most recent reports
#                                 without scanning all keys.
#   reports:record:<report_id>  — JSON string; TTL = _REPORT_RECORD_TTL.
#
# The index is capped at _REPORTS_MAX_SIZE entries (oldest pruned on write).
# ─────────────────────────────────────────────────────────────────────────────

import re
import time
import uuid

_REPORTS_INDEX_KEY = "reports:index"
_REPORTS_RECORD_PREFIX = "reports:record:"
_REPORT_RECORD_TTL = 30 * 24 * 3600   # 30 days
_REPORTS_MAX_SIZE = 100                # max entries kept in sorted set

# Words that carry little discriminating power for topic matching.
# Using this filtered set deliberately avoids false positives from
# shared meta-vocabulary (e.g. "analysis", "research", "impact").
_STOPWORDS: frozenset[str] = frozenset({
    # grammatical / functional
    "the", "and", "for", "are", "was", "were", "has", "have", "had",
    "its", "this", "that", "these", "those", "will", "would", "could",
    "should", "may", "might", "can", "but", "not", "all", "any",
    # question words
    "how", "what", "why", "when", "where", "which", "who",
    # common verbs
    "does", "did", "been", "being",
    # research meta-vocabulary — deliberately excluded to reduce false positives
    "analysis", "research", "study", "review", "overview", "impact",
    "effects", "effect", "role", "use", "used", "using",
})


def _tokenize(text: str) -> set[str]:
    """
    Return the significant tokens in *text* for keyword-overlap matching.

    Extracts 3+-character lowercase alphabetic runs, drops stopwords.
    Minimum length 3 avoids noise from 'is', 'at', 'of', 'ai', etc.

    NOTE: 'AI' and 'ML' (2 chars) are filtered out.  If your domain relies
    heavily on short acronyms, consider lowering the minimum to 2 and
    extending _STOPWORDS accordingly.
    """
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def save_report_record(
    *,
    session_id: str,
    query: str,
    final_report: str,
    sub_questions: list[str],
) -> str:
    """
    Persist a compact record of a completed report for cross-run lookup.

    Strips markdown headers from final_report and stores the first 500
    characters as an executive summary.

    Returns the generated report_id (UUID4 string).
    Raises redis.RedisError if the Redis connection is unavailable.
    """
    r = get_client()
    report_id = str(uuid.uuid4())
    now = time.time()

    # Strip markdown headers for a cleaner plain-text summary
    clean = re.sub(r"#+\s+[^\n]*\n?", "", final_report).strip()
    executive_summary = clean[:500]

    record = {
        "report_id": report_id,
        "session_id": session_id,
        "query": query,
        "sub_questions": sub_questions,
        "executive_summary": executive_summary,
        "timestamp": now,
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now)),
    }

    record_key = f"{_REPORTS_RECORD_PREFIX}{report_id}"
    r.set(record_key, json.dumps(record), ex=_REPORT_RECORD_TTL)
    r.zadd(_REPORTS_INDEX_KEY, {report_id: now})
    # Prune oldest entries beyond the cap
    r.zremrangebyrank(_REPORTS_INDEX_KEY, 0, -(_REPORTS_MAX_SIZE + 1))

    logger.info("Saved report record report_id=%s query=%r", report_id, query[:60])
    return report_id


def get_recent_reports(n: int = 20) -> list[dict]:
    """
    Return up to *n* most recent report records, newest first.

    Skips entries whose record key has expired (TTL elapsed but index not yet
    pruned — possible in long-lived Redis instances).
    """
    r = get_client()
    # ZREVRANGE returns members from highest score (newest) first.
    # Equivalent to ZRANGE ... REV but supported by all redis-py/fakeredis
    # versions, including fakeredis 2.x which lacks the `rev` keyword arg.
    report_ids = r.zrevrange(_REPORTS_INDEX_KEY, 0, n - 1)

    records: list[dict] = []
    for rid in report_ids:
        raw = r.get(f"{_REPORTS_RECORD_PREFIX}{rid}")
        if raw is None:
            continue  # record expired; index will be pruned on next write
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning("Corrupt report record for report_id=%s — skipped", rid)
    return records


def find_related_reports(
    query: str,
    n_recent: int = 20,
    min_overlap: int = 2,
) -> list[dict]:
    """
    Surface past reports whose topic overlaps with *query*.

    Matching strategy: token overlap after stopword filtering.
    The sub_questions stored with each past report are also tokenized and
    included in the comparison, giving broader coverage of the past topic.

    IMPORTANT LIMITATIONS — this is NOT semantic similarity search:

      False positives:
        Topics that share generic vocabulary (e.g. "drug", "cancer", "model")
        may match even though the actual research questions differ.

      False negatives:
        Semantically equivalent queries that use different words will NOT
        match (e.g. "ML in pharma" vs "machine learning pharmaceutical").
        Short acronyms (AI, ML) are excluded by the 3-char minimum and will
        never contribute to a match.

    Semantic search (cosine similarity over embeddings) was deliberately
    excluded from this architecture (no vector store in scope for Stage 5).
    A future upgrade could add Redis Stack RediSearch + vector field without
    changing this function's signature.

    Args:
        query:       the current research query to match against.
        n_recent:    how many recent reports to scan (default 20).
        min_overlap: minimum shared-token count to consider a match (default 2).

    Returns:
        List of matching report records (dict), sorted by overlap count
        descending then recency descending.  Each record has two extra keys
        injected for transparency:
          _overlap_tokens: sorted list of the shared tokens
          _overlap_count:  integer count of shared tokens
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    related: list[dict] = []
    for record in get_recent_reports(n_recent):
        # Build the past topic's token set from both query AND sub_questions.
        past_tokens = _tokenize(record.get("query", ""))
        for sq in record.get("sub_questions", []):
            past_tokens |= _tokenize(sq)

        overlap = query_tokens & past_tokens
        if len(overlap) >= min_overlap:
            record = dict(record)  # shallow copy — don't mutate stored record
            record["_overlap_tokens"] = sorted(overlap)
            record["_overlap_count"] = len(overlap)
            related.append(record)

    # Best matches first; recency as tiebreaker
    related.sort(key=lambda r: (-r["_overlap_count"], -r.get("timestamp", 0)))
    return related
