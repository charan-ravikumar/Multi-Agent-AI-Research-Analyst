"""
jobs/store.py — run records and the job stream.

A *run* is a hash (``<p>:run:<id>``): its status, inputs, progress, live draft and token usage.  A *job* is a message
on the stream ``<p>:jobs`` naming a run and a stage (``plan`` or ``research``), consumed by a consumer group.

  queued -> running -> awaiting_approval -> queued -> running -> done
                   \\-> failed / cancelled

Guarantees and how they are kept:
  * Every state change that involves the stream is ONE Lua script, so there is no moment where a run is marked
    queued without a job, or finished with its job still pending.
  * Delivery is at-least-once.  A job stays pending until it is acknowledged; if its worker stops heartbeating,
    ``XAUTOCLAIM`` hands it to another one.
  * A per-run lease (``<p>:lease:<id>``) with a random token fences the record: a worker that lost its lease can no
    longer beat, finish or release the run.  It does NOT fence checkpoint writes; the worker cancels its own run
    as soon as it learns the lease is gone.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

import redis

from research_analyst.config import settings

QUEUED, RUNNING, AWAITING_APPROVAL = "queued", "running", "awaiting_approval"
DONE, FAILED, CANCELLED = "done", "failed", "cancelled"
TERMINAL = frozenset({DONE, FAILED, CANCELLED})
PLAN, RESEARCH = "plan", "research"

OK, CANCEL, LOST = "ok", "cancel", "lost"        # answers to a heartbeat

_GROUP = "workers"
_JSON_FIELDS = ("approved", "progress", "usage")

# KEYS: run, stream      ARGV: ttl, run_id, now, query, review_rounds
_SUBMIT = """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1], 'run_id', ARGV[2], 'status', 'queued', 'stage', 'plan', 'created_at', ARGV[3],
           'query', ARGV[4], 'review_rounds', ARGV[5])
redis.call('EXPIRE', KEYS[1], ARGV[1])
redis.call('XADD', KEYS[2], '*', 'run_id', ARGV[2], 'stage', 'plan')
return 1
"""

# KEYS: run, stream      ARGV: ttl, run_id, approved_json
_APPROVE = """
if redis.call('HGET', KEYS[1], 'status') ~= 'awaiting_approval' then return 0 end
redis.call('HSET', KEYS[1], 'status', 'queued', 'stage', 'research', 'approved', ARGV[3])
redis.call('HDEL', KEYS[1], 'progress', 'draft', 'error')
redis.call('EXPIRE', KEYS[1], ARGV[1])
redis.call('XADD', KEYS[2], '*', 'run_id', ARGV[2], 'stage', 'research')
return 1
"""

# KEYS: run              ARGV: now
_CANCEL = """
local status = redis.call('HGET', KEYS[1], 'status')
if not status then return '' end
if status == 'queued' or status == 'awaiting_approval' then
  redis.call('HSET', KEYS[1], 'status', 'cancelled', 'finished_at', ARGV[1])
  return 'cancelled'
end
if status == 'running' then redis.call('HSET', KEYS[1], 'cancel', '1') end
return status
"""

# KEYS: run, lease, stream   ARGV: token, group, msg_id, stage, now, worker, ttl
_START = """
if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end
local status = redis.call('HGET', KEYS[1], 'status')
local stale = (not status) or status == 'done' or status == 'failed' or status == 'cancelled'
              or status == 'awaiting_approval' or redis.call('HGET', KEYS[1], 'stage') ~= ARGV[4]
if not stale and redis.call('HGET', KEYS[1], 'cancel') == '1' then
  redis.call('HSET', KEYS[1], 'status', 'cancelled', 'finished_at', ARGV[5])
  stale = true
end
if stale then
  redis.call('XACK', KEYS[3], ARGV[2], ARGV[3])
  redis.call('XDEL', KEYS[3], ARGV[3])
  redis.call('DEL', KEYS[2])
  return 0
end
redis.call('HSET', KEYS[1], 'status', 'running', 'worker', ARGV[6])
redis.call('HINCRBY', KEYS[1], 'attempts', 1)
if ARGV[4] == 'research' then redis.call('HSETNX', KEYS[1], 'started_at', ARGV[5]) end
redis.call('EXPIRE', KEYS[1], ARGV[7])
return 1
"""

# KEYS: run, lease, stream   ARGV: token, group, consumer, msg_id, lease_ms, ttl, field, value, ...
_BEAT = """
if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 'lost' end
redis.call('PEXPIRE', KEYS[2], ARGV[5])
redis.call('XCLAIM', KEYS[3], ARGV[2], ARGV[3], 0, ARGV[4], 'JUSTID')
if #ARGV > 6 then redis.call('HSET', KEYS[1], unpack(ARGV, 7)) end
redis.call('EXPIRE', KEYS[1], ARGV[6])
if redis.call('HGET', KEYS[1], 'cancel') == '1' then return 'cancel' end
return 'ok'
"""

# KEYS: run, lease, stream, dead   ARGV: token, group, msg_id, status, ttl, dead_flag, field, value, ...
_FINISH = """
if redis.call('GET', KEYS[2]) ~= ARGV[1] then return '' end
local status = ARGV[4]
if status == 'awaiting_approval' and redis.call('HGET', KEYS[1], 'cancel') == '1' then status = 'cancelled' end
redis.call('HSET', KEYS[1], 'status', status, unpack(ARGV, 7))
redis.call('EXPIRE', KEYS[1], ARGV[5])
if ARGV[6] == '1' then
  redis.call('XADD', KEYS[4], 'MAXLEN', '~', 1000, '*', 'run_id', redis.call('HGET', KEYS[1], 'run_id') or '',
             'msg_id', ARGV[3])
end
redis.call('XACK', KEYS[3], ARGV[2], ARGV[3])
redis.call('XDEL', KEYS[3], ARGV[3])
redis.call('DEL', KEYS[2])
return status
"""

# KEYS: run, lease, stream   ARGV: token, group, consumer, msg_id, idle_ms, deliveries
_RELEASE = """
if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end
redis.call('HSET', KEYS[1], 'status', 'queued')
redis.call('XCLAIM', KEYS[3], ARGV[2], ARGV[3], 0, ARGV[4], 'IDLE', ARGV[5], 'RETRYCOUNT', ARGV[6], 'JUSTID')
redis.call('DEL', KEYS[2])
return 1
"""


@dataclass(frozen=True)
class Job:
    msg_id: str
    run_id: str
    stage: str
    deliveries: int          # how many times the stream has handed this job to a worker (1 = first time)
    consumer: str
    token: str               # lease token proving this delivery owns the run


class RunStore:
    """All reads and writes of run records and jobs.  ``client`` must decode responses to str."""

    def __init__(self, client: redis.Redis, *, prefix: str | None = None, ttl_seconds: int | None = None) -> None:
        self.client = client
        self.prefix = prefix or settings.queue_namespace
        self.ttl = ttl_seconds or settings.run_ttl_seconds
        self.stream = f"{self.prefix}:jobs"
        self.dead_stream = f"{self.prefix}:jobs:dead"

    def run_key(self, run_id: str) -> str:
        return f"{self.prefix}:run:{run_id}"

    def lease_key(self, run_id: str) -> str:
        return f"{self.prefix}:lease:{run_id}"

    def _eval(self, script: str, keys: list[str], *args: Any) -> Any:
        return self.client.eval(script, len(keys), *keys, *args)

    # ── used by the UI ───────────────────────────────────────────────────────

    def submit(self, run_id: str, query: str, review_rounds: int = 0) -> bool:
        """Create the run and enqueue planning.  False if the run already exists (safe to call again)."""
        return bool(self._eval(_SUBMIT, [self.run_key(run_id), self.stream],
                               self.ttl, run_id, time.time(), query, review_rounds))

    def approve(self, run_id: str, sub_questions: list[str]) -> bool:
        """Enqueue research with the approved plan.  False unless the run is waiting for approval (no double start)."""
        return bool(self._eval(_APPROVE, [self.run_key(run_id), self.stream],
                               self.ttl, run_id, json.dumps(sub_questions)))

    def cancel(self, run_id: str) -> str:
        """Stop a run.  Waiting runs are cancelled at once; a running one is told through its next heartbeat."""
        return str(self._eval(_CANCEL, [self.run_key(run_id)], time.time()))

    def get(self, run_id: str) -> dict[str, Any] | None:
        raw = cast("dict[str, str]", self.client.hgetall(self.run_key(run_id)))
        if not raw:
            return None
        record: dict[str, Any] = dict(raw)
        for name in _JSON_FIELDS:
            if record.get(name):
                record[name] = json.loads(record[name])
        for name in ("review_rounds", "attempts"):
            record[name] = int(record.get(name) or 0)
        return record

    def workers_alive(self) -> int:
        return sum(1 for _ in self.client.scan_iter(match=f"{self.prefix}:worker:*", count=500))

    # ── used by workers ──────────────────────────────────────────────────────

    def ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.stream, _GROUP, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def mark_alive(self, consumer: str, ttl_seconds: int) -> None:
        self.client.set(f"{self.prefix}:worker:{consumer}", "1", ex=ttl_seconds)

    def forget_worker(self, consumer: str) -> None:
        self.client.delete(f"{self.prefix}:worker:{consumer}")

    def next_job(self, consumer: str, *, lease_ms: int, block_ms: int) -> Job | None:
        """An abandoned job (idle longer than a lease) if there is one, else the next new one, with its lease taken.
        Returns None when there is nothing to do or another worker still holds the run's lease."""
        claimed = cast("list[Any]", self.client.xautoclaim(
            self.stream, _GROUP, consumer, min_idle_time=lease_ms, start_id="0-0", count=1))
        entries = claimed[1]
        if not entries:
            read = cast("list[Any]", self.client.xreadgroup(
                _GROUP, consumer, {self.stream: ">"}, count=1, block=block_ms))
            entries = read[0][1] if read else []
        if not entries:
            return None
        msg_id, fields = entries[0]
        pending = cast("list[dict]", self.client.xpending_range(self.stream, _GROUP, min=msg_id, max=msg_id, count=1))
        if not pending:
            return None
        token = f"{consumer}:{uuid.uuid4().hex}"
        if not self.client.set(self.lease_key(fields["run_id"]), token, nx=True, px=lease_ms):
            return None                       # still owned; it stays pending and is offered again if the lease lapses
        return Job(msg_id, fields["run_id"], fields["stage"], int(pending[0]["times_delivered"]), consumer, token)

    def _keys(self, job: Job) -> list[str]:
        return [self.run_key(job.run_id), self.lease_key(job.run_id), self.stream]

    def start(self, job: Job) -> dict[str, Any] | None:
        """Mark the run as running.  None if the job is stale (finished, cancelled, wrong stage): it is dropped."""
        ok = self._eval(_START, self._keys(job), job.token, _GROUP, job.msg_id, job.stage, time.time(),
                        job.consumer, self.ttl)
        return self.get(job.run_id) if ok else None

    def beat(self, job: Job, *, lease_ms: int, fields: dict[str, str] | None = None) -> str:
        """Extend the lease, keep the job from being reclaimed, publish ``fields``.  Returns ok / cancel / lost."""
        flat = [x for pair in (fields or {}).items() for x in pair]
        return str(self._eval(_BEAT, self._keys(job), job.token, _GROUP, job.consumer, job.msg_id, lease_ms,
                              self.ttl, *flat))

    def finish(self, job: Job, status: str, fields: dict[str, str] | None = None, *, dead: bool = False) -> str:
        """Record the stage's outcome and acknowledge the job.  Returns the stored status, "" if the lease was lost."""
        merged = {"finished_at": str(time.time()), **(fields or {})}
        flat = [x for pair in merged.items() for x in pair]
        return str(self._eval(_FINISH, [*self._keys(job), self.dead_stream], job.token, _GROUP, job.msg_id,
                              status, self.ttl, "1" if dead else "0", *flat))

    def release(self, job: Job, *, lease_ms: int) -> bool:
        """Hand an unfinished job back (clean shutdown, Redis hiccup): immediately claimable, not counted as a try."""
        return bool(self._eval(_RELEASE, self._keys(job), job.token, _GROUP, job.consumer, job.msg_id,
                               lease_ms * 2, max(0, job.deliveries - 1)))
