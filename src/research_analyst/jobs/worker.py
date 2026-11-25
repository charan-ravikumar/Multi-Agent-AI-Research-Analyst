"""
jobs/worker.py — the worker process:  python -m research_analyst.jobs.worker

Each worker runs ``WORKER_CONCURRENCY`` slots on one long-lived event loop.  A slot claims a job, runs its stage
and records the outcome.  While it runs, a HEARTBEAT THREAD (not a coroutine, so a busy loop or a full executor
cannot starve it) extends the lease, publishes progress / live draft / token usage, and watches for cancellation.

How a stage can end:
  finished / failed     recorded, job acknowledged.  Application errors are NOT retried: the LLM layer already does.
  user pressed Stop     run cancelled, job acknowledged
  clean shutdown        job handed back at once, not counted as an attempt
  Redis unreachable     job handed back (or, if that fails too, reclaimed when the lease lapses)
  process killed        heartbeats stop; after one lease another worker claims the job and resumes
  lease lost            this worker abandons the run silently: someone else owns it now
A job delivered more than ``JOB_MAX_DELIVERIES`` times (it keeps killing workers) is failed and dead-lettered.

Free-tier note: the LLM rate limiter is still per process, so run ONE worker process until it is shared.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import redis

from research_analyst.config import settings
from research_analyst.jobs import store as S
from research_analyst.jobs.execution import execute_stage, run_config
from research_analyst.observability import runs, tracing
from research_analyst.observability.logger import get_logger
from research_analyst.observability.progress import progress
from research_analyst.observability.usage import tracker

_log = get_logger("worker")

_INFRA_ERRORS = (redis.ConnectionError, redis.TimeoutError)
_TIME_LIMIT_MESSAGE = "Run exceeded the time limit and was cancelled."
_SHUTDOWN = "shutdown"


def _live_fields(run_id: str) -> dict[str, str]:
    """Progress, draft and token usage of a run as run-record fields (only what currently exists)."""
    fields: dict[str, str] = {}
    snap = progress.snapshot(run_id)
    if snap:
        fields["draft"] = snap.pop("stream", "")
        fields["progress"] = json.dumps(snap)
    usage = tracker.export(run_id)
    if usage:
        fields["usage"] = json.dumps(usage)
    return fields


class _Heartbeat(threading.Thread):
    """Beats for one job until stopped; cancels the run's task when told to stop or when the lease is gone."""

    def __init__(self, worker: Worker, job: S.Job, loop: asyncio.AbstractEventLoop, task: asyncio.Task) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{job.run_id[:8]}")
        self.worker, self.job, self.loop, self.task = worker, job, loop, task
        self.verdict = S.OK
        self._halt = threading.Event()

    def run(self) -> None:
        w = self.worker
        last_ok = time.monotonic()
        while not self._halt.wait(w.heartbeat_s):
            try:
                verdict = w.store.beat(self.job, lease_ms=w.lease_ms, fields=_live_fields(self.job.run_id))
                last_ok = time.monotonic()
            except Exception as exc:  # noqa: BLE001 — a missed beat is survivable until the lease runs out
                _log.warning(f"heartbeat failed: {exc}", step="heartbeat_error", session_id=self.job.run_id)
                verdict = S.LOST if time.monotonic() - last_ok > w.lease_ms / 1000 else S.OK
            if verdict != S.OK and not self._halt.is_set():
                self.verdict = verdict
                self.loop.call_soon_threadsafe(self.task.cancel)
                return

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=15)


class Worker:
    def __init__(
        self, store: S.RunStore, graph: Any, *, consumer: str | None = None, concurrency: int | None = None,
        lease_ms: int | None = None, heartbeat_s: float | None = None, max_deliveries: int | None = None,
        block_ms: int = 1000,
    ) -> None:
        self.store, self.graph = store, graph
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.concurrency = concurrency or settings.worker_concurrency
        self.lease_ms = lease_ms or int(settings.job_lease_seconds * 1000)
        self.heartbeat_s = heartbeat_s or settings.job_heartbeat_seconds
        self.max_deliveries = max_deliveries or settings.job_max_deliveries
        self.block_ms = block_ms
        self._io = ThreadPoolExecutor(max_workers=self.concurrency + 1, thread_name_prefix="queue")

    # ── main loop ────────────────────────────────────────────────────────────

    async def run(self, stop: asyncio.Event) -> None:
        self.store.ensure_group()
        _log.info(f"worker {self.consumer} ready ({self.concurrency} slot(s))", step="worker_start")
        try:
            await asyncio.gather(self._announce(stop),
                                 *(self._slot(f"{self.consumer}-{i}", stop) for i in range(self.concurrency)))
        finally:
            with contextlib.suppress(Exception):
                self.store.forget_worker(self.consumer)
            _log.info(f"worker {self.consumer} stopped", step="worker_stop")

    async def _announce(self, stop: asyncio.Event) -> None:
        """Presence key, so the UI can say "no worker is running" instead of spinning forever."""
        ttl = max(5, int(self.lease_ms / 1000))
        while not stop.is_set():
            try:
                await asyncio.get_running_loop().run_in_executor(self._io, self.store.mark_alive, self.consumer, ttl)
            except Exception as exc:  # noqa: BLE001
                _log.warning(f"could not announce worker: {exc}", step="worker_announce_error")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), ttl / 3)

    async def _slot(self, name: str, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            try:
                job = await loop.run_in_executor(
                    self._io, lambda: self.store.next_job(name, lease_ms=self.lease_ms, block_ms=self.block_ms))
                if job is None:
                    await asyncio.sleep(0.05)
                    continue
                await self.process(job, stop)
            except _INFRA_ERRORS as exc:
                _log.warning(f"redis unavailable: {exc}", step="worker_redis_error")
                await asyncio.sleep(1.0)
            except redis.ResponseError as exc:
                if "NOGROUP" not in str(exc):              # the stream was flushed: recreate it and carry on
                    _log.error(f"redis error: {exc}", step="worker_redis_error")
                    await asyncio.sleep(1.0)
                self.store.ensure_group()
            except Exception as exc:  # noqa: BLE001 — one broken job must not take the slot down
                _log.error(f"worker slot error: {type(exc).__name__}: {exc}", step="worker_error")
                await asyncio.sleep(1.0)

    # ── one job ──────────────────────────────────────────────────────────────

    async def process(self, job: S.Job, stop: asyncio.Event) -> None:
        if job.deliveries > self.max_deliveries:
            message = f"Gave up after {self.max_deliveries} attempts: the worker running it kept stopping."
            self.store.finish(job, S.FAILED, {"error": message}, dead=True)
            _log.error(message, step="job_dead_letter", session_id=job.run_id)
            return
        record = self.store.start(job)
        if record is None:
            return
        run_id = job.run_id
        if record.get("usage"):
            tracker.restore(run_id, record["usage"])       # budget and totals carry over between stages and attempts
        _log.info(f"{job.stage} started (delivery {job.deliveries})", step="job_start", session_id=run_id)

        task = asyncio.create_task(self._execute(job.stage, record))
        beat = _Heartbeat(self, job, asyncio.get_running_loop(), task)
        beat.start()
        stopper = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if not task.done():
                task.cancel()
            status, state = await task
        except asyncio.CancelledError:
            beat.stop()
            self._cancelled(job, record, beat.verdict)
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise                                       # the worker itself is being torn down
        except _INFRA_ERRORS as exc:
            beat.stop()
            _log.warning(f"redis failed during the run, handing the job back: {exc}", step="job_requeued",
                         session_id=run_id)
            with contextlib.suppress(Exception):
                self.store.release(job, lease_ms=self.lease_ms)
        except Exception as exc:  # noqa: BLE001
            beat.stop()
            error = _TIME_LIMIT_MESSAGE if isinstance(exc, TimeoutError) else str(exc)
            self._complete(job, record, S.FAILED, None, error)
        else:
            beat.stop()
            self._complete(job, record, status, state, "")
        finally:
            stopper.cancel()
            if beat.is_alive():
                beat.stop()
            if not task.done():
                task.cancel()
            progress.pop(run_id)
            tracker.pop(run_id)

    async def _execute(self, stage: str, record: dict[str, Any]) -> tuple[str, dict | None]:
        run_id = record["run_id"]
        phase = "planning" if stage == S.PLAN else "research"
        tags = run_config(run_id, record["review_rounds"])["tags"]
        with tracing.run_span(phase, run_id, record.get("query", ""), tags=tags):
            status, state = await asyncio.wait_for(
                execute_stage(self.graph, stage, record), timeout=settings.run_timeout_seconds)
            if stage == S.RESEARCH:
                tracing.record_scores(state)
            return status, state

    def _cancelled(self, job: S.Job, record: dict[str, Any], verdict: str) -> None:
        if verdict == S.CANCEL:
            progress.cancel(job.run_id)
            self._complete(job, record, S.CANCELLED, None, "")
        elif verdict == S.LOST:
            _log.warning("lease lost; leaving the run to its new owner", step="job_lease_lost", session_id=job.run_id)
        else:
            with contextlib.suppress(Exception):
                self.store.release(job, lease_ms=self.lease_ms)
            _log.info("job handed back for another worker", step="job_released", session_id=job.run_id)

    def _complete(self, job: S.Job, record: dict[str, Any], status: str, state: dict | None, error: str) -> None:
        """Store the outcome (only if this worker still owns the run), then write the Monitor page's run record."""
        run_id = job.run_id
        fields = _live_fields(run_id)
        if status == S.DONE:
            progress.finish(run_id)
            fields = _live_fields(run_id)
            if state and state.get("usage_summary"):
                fields["summary"] = json.dumps(state["usage_summary"], default=str)
        fields["error"] = error
        usage = tracker.summary(run_id)
        stored = self.store.finish(job, status, fields)
        if not stored:
            _log.warning("lease lost before the result could be stored", step="job_lease_lost", session_id=run_id)
            return
        log = _log.error if stored == S.FAILED else _log.info
        log(f"{job.stage} -> {stored}" + (f": {error[:200]}" if error else ""), step="job_end", session_id=run_id)
        if job.stage == S.RESEARCH:
            started = float(record.get("started_at") or time.time())
            kind = runs.classify(state, error, stored == S.CANCELLED)
            runs.append(runs.build_record(
                run_id, kind, state, error, time.time() - started, record.get("query", ""), usage=usage,
                extra={"review_rounds": record["review_rounds"], "attempts": record["attempts"]}))


# ── process entry point ──────────────────────────────────────────────────────

def build_worker(**overrides: Any) -> Worker:
    from research_analyst.jobs.checkpoint import RedisCheckpointSaver  # noqa: PLC0415
    from research_analyst.jobs.connection import new_client  # noqa: PLC0415
    from research_analyst.pipeline.graph import build_graph  # noqa: PLC0415

    saver = RedisCheckpointSaver(new_client(decode=False), prefix=settings.queue_namespace,
                                 ttl_seconds=settings.run_ttl_seconds)
    return Worker(S.RunStore(new_client(decode=True)), build_graph(checkpointer=saver), **overrides)


async def serve(worker: Worker) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await worker.run(stop)


def main() -> None:
    if not settings.gemini_keys:
        raise SystemExit("GEMINI_API_KEY is not set")
    worker = build_worker()
    try:
        worker.store.client.ping()
    except redis.RedisError as exc:
        raise SystemExit(f"Cannot reach Redis (check REDIS_URL): {type(exc).__name__}: {exc}") from exc
    asyncio.run(serve(worker))


if __name__ == "__main__":
    main()
