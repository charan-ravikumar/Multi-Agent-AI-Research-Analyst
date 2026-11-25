"""kill -9 a real worker process mid-run; a second process finishes the run.  Needs a real Redis."""
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import pytest

from research_analyst.jobs import store as S
from research_analyst.jobs.checkpoint import RedisCheckpointSaver
from research_analyst.pipeline.graph import build_graph
from tests import redis_backends

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def real(real_redis_url):
    if not real_redis_url:
        pytest.skip("no redis-server available")
    backend = redis_backends.real_backend(real_redis_url)
    yield backend
    redis_backends.cleanup(backend)


def _wait(store, run_id, predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get(run_id)
        if record and predicate(record):
            return record
        time.sleep(0.05)
    raise AssertionError(f"timed out; last record: {store.get(run_id)}")


def test_a_killed_worker_process_is_replaced_and_finished_work_is_not_repeated(real, tmp_path):
    store = S.RunStore(real.text, prefix=real.prefix, ttl_seconds=300)
    calls = tmp_path / "calls.log"
    env = {**os.environ, "REDIS_URL": real.url, "QUEUE_NAMESPACE": real.prefix, "FAKE_CALL_LOG": str(calls),
           "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]), "JOB_LEASE_SECONDS": "1.5",
           "JOB_HEARTBEAT_SECONDS": "0.1", "WORKER_CONCURRENCY": "1", "LOG_FILE": str(tmp_path / "trace.jsonl"),
           "RUNS_FILE": str(tmp_path / "runs.jsonl")}
    processes = []

    def start(**extra):
        proc = subprocess.Popen([sys.executable, "-m", "tests.fake_worker"], cwd=tmp_path, env={**env, **extra},
                                stdout=subprocess.DEVNULL, stderr=(tmp_path / f"worker{len(processes)}.err").open("w"))
        processes.append(proc)
        return proc

    try:
        run_id = str(uuid.uuid4())
        store.submit(run_id, "What is X?")
        first = start(FAKE_HANG_ASPECT="aspect 3")
        _wait(store, run_id, lambda r: r["status"] == S.AWAITING_APPROVAL)
        graph = build_graph(checkpointer=RedisCheckpointSaver(real.binary, prefix=real.prefix))
        sqs = graph.get_state({"configurable": {"thread_id": run_id}}).values["sub_questions"]
        store.approve(run_id, sqs)
        _wait(store, run_id, lambda r: (r.get("progress") or {}).get("branches_done") == 3)

        first.send_signal(signal.SIGKILL)
        first.wait(timeout=10)
        second = start()
        record = _wait(store, run_id, lambda r: r["status"] in S.TERMINAL)
        assert record["status"] == S.DONE, record.get("error")

        second.send_signal(signal.SIGTERM)                      # a clean stop exits promptly
        assert second.wait(timeout=20) == 0
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()

    made = [line.split("|") for line in calls.read_text().splitlines()]
    synth = Counter((aspect, int(pid)) for pid, role, aspect in made if role == "synth")
    assert dict(list(synth)) == {                               # which process wrote which section
        "aspect 0": first.pid, "aspect 1": first.pid, "aspect 2": first.pid, "aspect 3": second.pid}
    assert sum(synth.values()) == 4                             # every section written exactly once
    assert [role for _, role, _ in made].count("planner") == 1
    state = graph.get_state({"configurable": {"thread_id": run_id}}).values
    assert state["final_report"] and len(state["draft"]) == 4
    assert real.text.xlen(store.stream) == 0
