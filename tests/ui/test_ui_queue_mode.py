"""Headless Streamlit in EXECUTION_MODE=queue: the UI submits and watches; a worker (driven here) does the work."""
import asyncio
import os
import uuid

import pytest
import redis

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from research_analyst.config import settings  # noqa: E402
from research_analyst.jobs import store as S  # noqa: E402
from research_analyst.jobs.checkpoint import RedisCheckpointSaver  # noqa: E402
from research_analyst.jobs.worker import Worker  # noqa: E402
from research_analyst.llm import preflight  # noqa: E402
from research_analyst.pipeline.graph import build_graph  # noqa: E402
from research_analyst.ui import queued  # noqa: E402
from tests import redis_backends  # noqa: E402
from tests.helpers import FakeLLM, make_search  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "..", "app", "main.py")


@pytest.fixture
def rig(monkeypatch):
    backend = redis_backends.fake_backend()
    store = S.RunStore(backend.text, prefix=backend.prefix, ttl_seconds=300)
    store.ensure_group()
    graph = build_graph(checkpointer=RedisCheckpointSaver(backend.binary, prefix=backend.prefix))
    monkeypatch.setattr(settings, "execution_mode", "queue")
    monkeypatch.setattr(queued, "get_store", lambda: store)
    monkeypatch.setattr(queued, "get_graph", lambda: graph)
    monkeypatch.setattr("research_analyst.llm.generate", FakeLLM())
    monkeypatch.setattr("research_analyst.agents.searcher.search_web", make_search())
    monkeypatch.setattr(preflight, "check_models", lambda force=False: preflight.ModelCheck(
        verified=True, available=(), missing=(), suggestions=()))

    def work():
        """One worker handles the next job, then stops."""
        async def one():
            worker = Worker(store, graph, consumer="ui-test", heartbeat_s=0.05, lease_ms=5000, block_ms=20)
            job = store.next_job("ui-test", lease_ms=5000, block_ms=20)
            assert job is not None
            await worker.process(job, asyncio.Event())
        asyncio.run(one())

    return store, work


def _app():
    return AppTest.from_file(APP, default_timeout=60)


def _start(at):
    at.run()
    at.text_input(key="query_input").set_value("What is X?")
    at.run()
    next(b for b in at.button if b.label.startswith("Start research")).click()
    at.run()
    return at.session_state["session_id"]


def test_the_whole_flow_goes_through_the_queue_and_survives_a_reload(rig):
    store, work = rig
    at = _app()
    run_id = _start(at)
    assert at.session_state["phase"] == "planning" and at.query_params["run"] == [run_id]
    assert store.get(run_id)["status"] == S.QUEUED
    assert any("No worker is running" in w.value for w in at.warning)

    work()                                              # planning
    at.run()
    assert at.session_state["phase"] == "awaiting_approval" and "aspect 0" in at.text_area(key="sq_textarea").value
    next(b for b in at.button if b.label.startswith("Approve")).click()
    at.run()
    assert at.session_state["phase"] == "researching" and store.get(run_id)["stage"] == S.RESEARCH
    assert "bg_thread" not in at.session_state

    work()                                              # research
    # AppTest keeps the approval widgets of the run that ended in st.rerun() and reads their (cleared) state again.
    at.session_state["sq_textarea"] = ""
    at.run()
    assert not at.exception and at.session_state["phase"] == "done"
    assert any("Executive Summary" in m.value for m in at.markdown)

    reloaded = _app()                                   # a new browser session on the same link
    reloaded.query_params["run"] = run_id
    reloaded.run()
    assert not reloaded.exception and reloaded.session_state["phase"] == "done"
    next(b for b in reloaded.button if b.label == "New research").click()
    reloaded.run()
    assert reloaded.session_state["phase"] == "idle" and "run" not in reloaded.query_params


def test_stop_cancels_the_queued_run(rig):
    store, _ = rig
    at = _app()
    run_id = _start(at)
    next(b for b in at.button if b.key == "stop_research").click()
    at.run()
    at.run()
    assert at.session_state["phase"] == "cancelled" and store.get(run_id)["status"] == S.CANCELLED


def test_a_reload_while_queued_returns_to_the_waiting_page(rig):
    store, _ = rig
    run_id = str(uuid.uuid4())
    store.submit(run_id, "What is X?")
    at = _app()
    at.query_params["run"] = run_id
    at.run()
    assert not at.exception and at.session_state["phase"] == "planning"


@pytest.mark.parametrize("value", ["not-a-run-id", str(uuid.UUID(int=1))])
def test_a_bad_or_expired_link_lands_on_the_start_page(rig, value):
    at = _app()
    at.query_params["run"] = value
    at.run()
    assert not at.exception and at.session_state["phase"] == "idle" and "run" not in at.query_params


def test_elapsed_time_covers_every_attempt_not_just_the_last():
    record = {"status": S.DONE, "started_at": "100.0", "finished_at": "167.5", "progress": {"elapsed_s": 35.0}}
    assert queued._elapsed_s(record) == 67.5 and queued._elapsed_s({"status": S.QUEUED}) is None
    assert queued._elapsed_s({"status": S.RUNNING, "started_at": "0"}) > 1000


def test_a_failed_run_shows_its_error(rig):
    store, _ = rig
    run_id = str(uuid.uuid4())
    store.submit(run_id, "What is X?")
    store.client.hset(store.run_key(run_id), mapping={"status": S.FAILED, "error": "boom happened"})
    at = _app()
    at.query_params["run"] = run_id
    at.run()
    assert at.session_state["phase"] == "error" and any("boom happened" in e.value for e in at.error)


def test_unreachable_redis_is_reported_not_raised(rig, monkeypatch):
    store, _ = rig

    def down(*a, **k):
        raise redis.ConnectionError("down")

    monkeypatch.setattr(store, "submit", down)
    at = _app()
    _start(at)
    assert not at.exception and any("Cannot reach Redis" in e.value for e in at.error)
