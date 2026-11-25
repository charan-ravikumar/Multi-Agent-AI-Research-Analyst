"""The Redis checkpointer against LangGraph's official conformance suite, plus what this project relies on."""
import asyncio

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.conformance import checkpointer_test, validate

from research_analyst.jobs.checkpoint import RedisCheckpointSaver
from research_analyst.models.fact import Fact
from research_analyst.models.research_plan import ResearchPlan
from tests import redis_backends


def test_passes_the_langgraph_conformance_suite_for_every_base_capability(backend):
    @checkpointer_test(name=f"RedisCheckpointSaver-{backend.name}")
    async def factory():
        redis_backends.cleanup(backend)   # the suite lists across all threads: start with this prefix empty
        yield RedisCheckpointSaver(backend.binary, prefix=backend.prefix)

    report = asyncio.run(validate(factory))
    failures = {cap: r.failures for cap, r in report.results.items() if r.failures}
    assert not failures, failures
    assert report.passed_all_base()


def _config(thread="t1"):
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}


def test_project_models_survive_a_round_trip_as_objects(backend):
    saver = RedisCheckpointSaver(backend.binary, prefix=backend.prefix)
    fact = Fact(sub_question="q", content="c", source_url="https://a.example", source_title="A", confidence=0.5)
    plan = ResearchPlan(query="q", sub_questions=[f"sub question number {i}?" for i in range(4)])
    cp = empty_checkpoint()
    cp["channel_values"] = {"extracted_facts": [fact], "research_plan": plan}
    cp["channel_versions"] = {"extracted_facts": "1", "research_plan": "1"}
    saver.put(_config(), cp, {"source": "input", "step": 0}, cp["channel_versions"])
    got = saver.get_tuple(_config()).checkpoint["channel_values"]
    assert got["extracted_facts"] == [fact] and isinstance(got["research_plan"], ResearchPlan)


def test_every_key_expires_and_deleting_a_thread_leaves_nothing(backend):
    saver = RedisCheckpointSaver(backend.binary, prefix=backend.prefix, ttl_seconds=120)
    cp = empty_checkpoint()
    cp["channel_values"], cp["channel_versions"] = {"a": 1}, {"a": "1"}
    saved = saver.put(_config(), cp, {"source": "input", "step": 0}, {"a": "1"})
    saver.put_writes(saved, [("a", 2)], task_id="task")
    keys = list(backend.binary.scan_iter(match=f"{backend.prefix}:*"))
    assert keys and all(0 < backend.binary.ttl(k) <= 120 for k in keys)
    saver.delete_thread("t1")
    left = [k for k in backend.binary.scan_iter(match=f"{backend.prefix}:*") if not k.endswith(b":threads")]
    assert left == [] and saver.get_tuple(_config()) is None
