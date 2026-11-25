"""Durable execution: store, worker and resume, on fakeredis and (when available) a real Redis.  LLM and search are fakes."""
import asyncio
import threading
import time
import uuid

import pytest
import redis

from research_analyst.config import settings
from research_analyst.jobs import store as S
from research_analyst.jobs.checkpoint import RedisCheckpointSaver
from research_analyst.jobs.worker import Worker
from research_analyst.observability import runs
from research_analyst.pipeline.graph import build_graph
from tests.helpers import FakeLLM, make_search

LEASE_MS = 600


class Rig:
    """A store, a checkpointer and workers sharing one Redis, with a scripted LLM."""

    def __init__(self, backend, monkeypatch, fake: FakeLLM | None = None):
        self.backend = backend
        self.fake = fake or FakeLLM()
        monkeypatch.setattr("research_analyst.llm.generate", self.fake)
        monkeypatch.setattr("research_analyst.agents.searcher.search_web", make_search())
        self.store = self.new_store()
        self.store.ensure_group()
        self.graph = build_graph(checkpointer=RedisCheckpointSaver(backend.binary, prefix=backend.prefix))

    def new_store(self) -> S.RunStore:
        return S.RunStore(self.backend.text, prefix=self.backend.prefix, ttl_seconds=300)

    def worker(self, name="w1", store=None, **kw) -> Worker:
        kw.setdefault("lease_ms", LEASE_MS)
        return Worker(store or self.new_store(), self.graph, consumer=name, heartbeat_s=0.05, block_ms=20, **kw)

    async def drain(self, run_id: str, until: set[str], worker: Worker | None = None, timeout: float = 30) -> dict:
        """Run a worker until the run reaches one of ``until``."""
        worker = worker or self.worker()
        stop = asyncio.Event()
        serving = asyncio.create_task(worker.run(stop))
        try:
            return await self.wait(run_id, lambda r: r["status"] in until, timeout)
        finally:
            stop.set()
            await serving

    async def wait(self, run_id: str, predicate, timeout: float = 30) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.store.get(run_id)
            if record and predicate(record):
                return record
            await asyncio.sleep(0.02)
        raise AssertionError(f"timed out; last record: {self.store.get(run_id)}")

    def submit_and_plan(self) -> tuple[str, list[str]]:
        run_id = str(uuid.uuid4())
        assert self.store.submit(run_id, "What is X?")
        asyncio.run(self.drain(run_id, {S.AWAITING_APPROVAL, S.FAILED}))
        snapshot = self.graph.get_state({"configurable": {"thread_id": run_id}})
        return run_id, snapshot.values["sub_questions"]

    def pending(self) -> int:
        return self.backend.text.xpending(self.store.stream, "workers")["pending"]


class Gate:
    """Makes the fact extraction of ONE sub-question hang until released (only the first time it is asked)."""

    def __init__(self, marker: str = "aspect 3"):
        self.marker, self.release, self.hits = marker, threading.Event(), 0

    def __call__(self, prompt, n):
        if self.marker in prompt[-1]["content"]:
            self.hits += 1
            if self.hits == 1:
                self.release.wait(20)
        return '{"facts": [{"source": 1, "content": "A fact.", "confidence": 0.9}]}'


# ── the normal path ───────────────────────────────────────────────────────────

def test_a_run_is_planned_approved_and_researched_by_a_worker(backend, monkeypatch):
    rig = Rig(backend, monkeypatch)
    run_id, sqs = rig.submit_and_plan()
    assert len(sqs) == 4 and rig.store.get(run_id)["status"] == S.AWAITING_APPROVAL and rig.pending() == 0

    assert rig.store.approve(run_id, sqs[:3])                       # the human drops one question
    record = asyncio.run(rig.drain(run_id, {S.DONE, S.FAILED}))

    assert record["status"] == S.DONE, record.get("error")
    state = rig.graph.get_state({"configurable": {"thread_id": run_id}}).values
    assert state["final_report"] and state["sub_questions"] == sqs[:3]
    assert rig.fake.calls["planner"] == 1 and rig.fake.calls["synth"] == 3
    assert record["progress"]["finished"] and record["progress"]["fraction"] == 1.0
    assert backend.text.xlen(rig.store.stream) == 0 and not backend.text.exists(rig.store.lease_key(run_id))
    [logged] = runs.load()
    assert logged["run_id"] == run_id and logged["status"] == "complete" and logged["attempts"] == 2
    # tokens of the planning stage are carried into the research stage's totals (fake LLM: 2 tokens per call)
    assert logged["tokens"] == 2 * rig.fake.total and logged["calls"] == rig.fake.total


def test_submitting_or_approving_twice_enqueues_only_once(backend, monkeypatch):
    rig = Rig(backend, monkeypatch)
    run_id, sqs = rig.submit_and_plan()
    assert not rig.store.submit(run_id, "What is X?")
    assert rig.store.approve(run_id, sqs) and not rig.store.approve(run_id, sqs)
    assert backend.text.xlen(rig.store.stream) == 1


def test_an_application_error_fails_the_run_without_retrying(backend, monkeypatch):
    def broken(prompt, n):
        raise RuntimeError("LLM call failed on model after retries")

    rig = Rig(backend, monkeypatch, FakeLLM(overrides={"synth": broken}))
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)
    record = asyncio.run(rig.drain(run_id, {S.DONE, S.FAILED}))
    assert record["status"] == S.FAILED and "branch(es) failed" in record["error"]
    assert rig.pending() == 0 and runs.load()[0]["status"] == "failed"


def test_an_expired_checkpoint_gives_a_clear_failure(backend, monkeypatch):
    rig = Rig(backend, monkeypatch)
    run_id, sqs = rig.submit_and_plan()
    rig.graph.checkpointer.delete_thread(run_id)
    rig.store.approve(run_id, sqs)
    record = asyncio.run(rig.drain(run_id, {S.DONE, S.FAILED}))
    assert record["status"] == S.FAILED and "expired" in record["error"]


def test_a_plan_job_redelivered_after_planning_does_not_plan_again(backend, monkeypatch):
    rig = Rig(backend, monkeypatch)
    run_id, _ = rig.submit_and_plan()
    backend.text.hset(rig.store.run_key(run_id), mapping={"status": S.QUEUED})     # died before reporting
    backend.text.xadd(rig.store.stream, {"run_id": run_id, "stage": S.PLAN})
    asyncio.run(rig.drain(run_id, {S.AWAITING_APPROVAL, S.FAILED}))
    assert rig.fake.calls["planner"] == 1


# ── cancellation ──────────────────────────────────────────────────────────────

def test_cancelling_a_waiting_run_drops_its_job(backend, monkeypatch):
    rig = Rig(backend, monkeypatch)
    run_id = str(uuid.uuid4())
    rig.store.submit(run_id, "What is X?")
    assert rig.store.cancel(run_id) == S.CANCELLED

    async def scenario():
        stop = asyncio.Event()
        serving = asyncio.create_task(rig.worker().run(stop))
        await asyncio.sleep(0.3)
        stop.set()
        await serving

    asyncio.run(scenario())
    assert rig.fake.total == 0 and backend.text.xlen(rig.store.stream) == 0
    assert rig.store.cancel("no-such-run") == "" and not rig.store.approve(run_id, ["q"])


def test_stop_cancels_a_running_run(backend, monkeypatch):
    gate = Gate()
    rig = Rig(backend, monkeypatch, FakeLLM(overrides={"reader": gate}))
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)

    async def scenario():
        stop = asyncio.Event()
        serving = asyncio.create_task(rig.worker().run(stop))
        await rig.wait(run_id, lambda r: (r.get("progress") or {}).get("branches_done") == 3)
        assert rig.store.cancel(run_id) == S.RUNNING
        record = await rig.wait(run_id, lambda r: r["status"] == S.CANCELLED)
        gate.release.set()
        stop.set()
        await serving
        return record

    asyncio.run(scenario())
    assert "writer" not in rig.fake.calls and rig.pending() == 0 and runs.load()[0]["status"] == "cancelled"


# ── crashes, hand-over, poison jobs ───────────────────────────────────────────

async def _crash_mid_research(rig: Rig, run_id: str, gate: Gate) -> None:
    """Worker 'dead' reaches the hung branch, then disappears: no more beats, and it never releases its job."""
    dead_store = rig.new_store()
    dead_store.release = lambda *a, **k: False                   # a killed process cannot hand anything back
    task = asyncio.create_task(rig.worker("dead", store=dead_store).run(asyncio.Event()))
    await rig.wait(run_id, lambda r: (r.get("progress") or {}).get("branches_done") == 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.store.get(run_id)["status"] == S.RUNNING and rig.pending() == 1


def test_another_worker_resumes_a_crashed_run_without_repeating_finished_branches(backend, monkeypatch):
    gate = Gate()
    rig = Rig(backend, monkeypatch, FakeLLM(overrides={"reader": gate}))
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)

    async def scenario():
        await _crash_mid_research(rig, run_id, gate)
        assert rig.fake.calls["synth"] == 3 and rig.fake.calls["reader"] == 4
        record = await rig.drain(run_id, {S.DONE, S.FAILED}, rig.worker("alive"))
        gate.release.set()
        return record

    record = asyncio.run(scenario())
    assert record["status"] == S.DONE, record.get("error")
    assert rig.fake.calls["synth"] == 4                 # only the interrupted branch ran again
    assert rig.fake.calls["reader"] == 5 and rig.fake.calls["planner"] == 1
    assert record["attempts"] == 3                      # plan, research (died), research (resumed)
    state = rig.graph.get_state({"configurable": {"thread_id": run_id}}).values
    assert len(state["draft"]) == 4 and rig.pending() == 0
    # usage of the dead worker's finished branches survived: 1 planner + 3*2 branch calls + the resumed work
    assert state["usage_summary"]["by_agent"]["synthesizer"]["calls"] == 4


def test_a_job_that_keeps_killing_workers_is_dead_lettered(backend, monkeypatch):
    gate = Gate()
    rig = Rig(backend, monkeypatch, FakeLLM(overrides={"reader": gate}))
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)

    async def scenario():
        await _crash_mid_research(rig, run_id, gate)
        record = await rig.drain(run_id, {S.DONE, S.FAILED}, rig.worker("alive", max_deliveries=1))
        gate.release.set()
        return record

    record = asyncio.run(scenario())
    assert record["status"] == S.FAILED and "Gave up" in record["error"]
    [(_, dead)] = backend.text.xrange(rig.store.dead_stream)
    assert dead["run_id"] == run_id and rig.pending() == 0


def test_a_clean_shutdown_hands_the_job_over_at_once_without_counting_an_attempt(backend, monkeypatch):
    gate = Gate()
    rig = Rig(backend, monkeypatch, FakeLLM(overrides={"reader": gate}))
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)

    async def scenario():
        stop = asyncio.Event()
        leaving = asyncio.create_task(rig.worker("leaving", lease_ms=60_000).run(stop))
        await rig.wait(run_id, lambda r: (r.get("progress") or {}).get("branches_done") == 3)
        stop.set()
        await leaving
        assert rig.store.get(run_id)["status"] == S.QUEUED
        job = rig.new_store().next_job("next", lease_ms=60_000, block_ms=20)      # no waiting for the lease
        assert job is not None and job.deliveries == 1
        assert rig.store.release(job, lease_ms=60_000)
        record = await rig.drain(run_id, {S.DONE, S.FAILED}, rig.worker("next", lease_ms=60_000))
        gate.release.set()
        return record

    assert asyncio.run(scenario())["status"] == S.DONE


def test_a_worker_that_lost_its_lease_cannot_write_and_stops(backend, monkeypatch):
    gate = Gate()
    rig = Rig(backend, monkeypatch, FakeLLM(overrides={"reader": gate}))
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)

    async def scenario():
        stop = asyncio.Event()
        serving = asyncio.create_task(rig.worker("zombie").run(stop))
        await rig.wait(run_id, lambda r: (r.get("progress") or {}).get("branches_done") == 3)
        backend.text.set(rig.store.lease_key(run_id), "someone-else", px=60_000)
        await asyncio.sleep(0.4)
        gate.release.set()
        await asyncio.sleep(0.2)
        stop.set()
        await serving

    asyncio.run(scenario())
    assert rig.store.get(run_id)["status"] == S.RUNNING          # untouched by the old owner
    assert "writer" not in rig.fake.calls and rig.pending() == 1
    stale = S.Job("0-0", run_id, S.RESEARCH, 1, "zombie", "old-token")
    assert rig.store.finish(stale, S.DONE) == "" and rig.store.beat(stale, lease_ms=100) == S.LOST
    assert not rig.store.release(stale, lease_ms=100) and rig.store.start(stale) is None


def test_a_redis_failure_during_a_run_hands_the_job_back(backend, monkeypatch):
    rig = Rig(backend, monkeypatch)
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)
    saver = rig.graph.checkpointer
    real_put, failures = saver.put, [1]

    def flaky(*args, **kwargs):
        if failures:
            failures.pop()
            raise redis.ConnectionError("connection reset")
        return real_put(*args, **kwargs)

    monkeypatch.setattr(saver, "put", flaky)
    record = asyncio.run(rig.drain(run_id, {S.DONE, S.FAILED}))
    assert record["status"] == S.DONE, record.get("error")
    assert not failures


def test_the_run_time_limit_fails_the_run(backend, monkeypatch):
    gate = Gate()
    rig = Rig(backend, monkeypatch, FakeLLM(overrides={"reader": gate}))
    run_id, sqs = rig.submit_and_plan()
    rig.store.approve(run_id, sqs)
    monkeypatch.setattr(settings, "run_timeout_seconds", 0.5)

    async def scenario():
        record = await rig.drain(run_id, {S.DONE, S.FAILED})
        gate.release.set()
        return record

    assert "time limit" in asyncio.run(scenario())["error"]


def test_worker_presence_is_visible_while_it_runs(backend, monkeypatch):
    rig = Rig(backend, monkeypatch)

    async def scenario():
        stop = asyncio.Event()
        serving = asyncio.create_task(rig.worker().run(stop))
        await asyncio.sleep(0.2)
        during = rig.store.workers_alive()
        stop.set()
        await serving
        return during

    assert asyncio.run(scenario()) == 1 and rig.store.workers_alive() == 0
