import asyncio
import json
import uuid

from core.gaps import COVERAGE, OPEN_QUESTION, curate_gaps, gap_texts, make_item
from core.progress import ProgressTracker, progress
from tests.helpers import FakeLLM, initial_state, make_search

N = 4


# ── gap curation ──────────────────────────────────────────────────────────────

def test_coverage_is_always_kept_and_open_questions_are_capped_and_deduped():
    items = [make_item("sq1", COVERAGE, "No search results for sq1"), make_item("sq1", COVERAGE, "No search results for sq1")]
    distinct = {
        "sq1": ["How does dosing affect outcomes in elderly patients?", "Which regulators approve hospital deployment internationally?"],
        "sq2": ["What are the long-term manufacturing costs per patient?", "How reliable are open benchmark datasets for validation?"],
        "sq3": ["Which privacy safeguards protect patient records during training?", "How do clinicians trust automated summaries day to day?"],
    }
    for sq, texts in distinct.items():
        items += [make_item(sq, OPEN_QUESTION, t) for t in texts]
    c = curate_gaps(items, max_open=4)
    assert len(c["coverage"]) == 1                                   # exact duplicate removed, none dropped
    assert len(c["open_questions"]) == 4 and c["dropped"] == 2
    assert {o["sub_question"] for o in c["open_questions"]} == {"sq1", "sq2", "sq3"}   # round-robin: every sub-question represented


def test_similar_open_questions_are_merged_across_sub_questions():
    items = [make_item("a", OPEN_QUESTION, "What is the long-term safety of CRISPR gene editing therapies?"),
             make_item("b", OPEN_QUESTION, "Long-term safety of CRISPR gene editing therapies?"),
             make_item("b", OPEN_QUESTION, "How much do these treatments cost per patient?")]
    c = curate_gaps(items, max_open=6)
    assert len(c["open_questions"]) == 2 and c["dropped"] == 1


def test_gap_texts_labels_open_questions():
    c = curate_gaps([make_item("a", COVERAGE, "Branch failed"), make_item("a", OPEN_QUESTION, "What about cost?")])
    assert gap_texts(c) == ["Branch failed", "Open question: What about cost?"]


def _many_gaps(prompt, n):
    return json.dumps({"synthesis": "Machine learning shortens early drug screening cycles. [F1]", "contradictions": [],
                       "gaps": [f"What is the effect of factor number {n} aspect {i} on clinical outcomes overall?" for i in range(5)]})


def test_synthesizer_keeps_at_most_two_model_gaps_and_graph_curates_for_the_writer(run_graph):
    fake = FakeLLM(N, {"synth": _many_gaps})
    res = run_graph(fake, make_search(2))
    per_sq = {}
    for g in res["gap_items"]:
        per_sq.setdefault(g["sub_question"], []).append(g)
    assert all(len(v) <= 2 for v in per_sq.values())
    writer_prompt = [p for r, p in fake.prompts if r == "writer"][0][1]["content"]
    assert 0 < writer_prompt.count("Open question:") <= 6              # curated, not all 8


def test_system_detected_problems_become_coverage_items(run_graph):
    fake = FakeLLM(N, {"synth": lambda p, n: '{"synthesis": "cut off'})                    # synthesis fails twice
    res = run_graph(fake, make_search(2))
    cov = [g for g in res["gap_items"] if g["kind"] == COVERAGE]
    assert len(cov) == N and all("Synthesis generation failed" in g["text"] for g in cov)


def test_empty_evidence_is_a_coverage_item(run_graph):
    fake = FakeLLM(N, {"reader": lambda p, n: json.dumps({"facts": []})})
    res = run_graph(fake, make_search(2))
    assert any(g["kind"] == COVERAGE and "No usable evidence" in g["text"] for g in res["gap_items"])


def test_failed_branch_is_a_coverage_item(run_graph):
    def reader(prompt, n):
        if "aspect 1" in prompt[1]["content"]:
            raise RuntimeError("boom")
        return FakeLLM(N)._default("reader")

    res = run_graph(FakeLLM(N, {"reader": reader}), make_search(2))
    assert any(g["kind"] == COVERAGE and "failed" in g["text"] for g in res["gap_items"])


# ── progress tracker ──────────────────────────────────────────────────────────

def test_tracker_units_and_failed_branch_accounting():
    t = ProgressTracker()
    t.start("s", n_branches=2, review_rounds=0)               # 6 branch units + 3 writing = 9
    b1, b2 = t.branch("s"), t.branch("s")
    b1.step("searched")
    assert 0 < t.snapshot("s")["fraction"] < 0.2
    b1.step("read")
    b1.step("wrote")
    b1.close()
    b2.close("branch failed")                                 # failed early: remaining 3 units counted once
    b2.close()
    snap = t.snapshot("s")
    assert snap["branches_done"] == 2 and abs(snap["fraction"] - 6 / 9) < 1e-9
    t.advance("s", "writing", 2)
    assert t.snapshot("s")["fraction"] < 1.0                  # never 100% before finish()
    t.finish("s")
    assert t.snapshot("s")["fraction"] == 1.0 and t.snapshot("s")["finished"]


def test_review_units_and_flush():
    t = ProgressTracker()
    t.start("s", 1, review_rounds=2)                          # review = 3 units
    assert t.snapshot("s")["fraction"] == 0
    t.advance("s", "review", 1)
    t.finish_stage("s", "review")                             # loop ended early (organic resolution)
    assert abs(t.snapshot("s")["fraction"] - 3 / 9) < 1e-9


def test_unknown_sessions_and_stages_are_ignored_and_sessions_bounded():
    t = ProgressTracker()
    t.advance("nope", "branches", 1)
    t.branch("nope").step("x")
    t.finish("nope")
    assert t.snapshot("nope") is None
    for i in range(300):
        t.start(str(i), 1)
    assert len(t._s) == 200


def test_progress_is_monotonic_and_completes_through_a_real_graph_run(monkeypatch):
    from orchestrator.graph import app
    sid = str(uuid.uuid4())
    seen: list[float] = []
    fake = FakeLLM(N)

    def spying(*a, **kw):
        snap = progress.snapshot(sid)
        seen.append(snap["fraction"] if snap else -1)
        return fake(*a, **kw)

    monkeypatch.setattr("llm.generate", spying)
    monkeypatch.setattr("agents.searcher.search_web", make_search(2))
    st = initial_state()
    st["session_id"] = sid
    progress.start(sid, n_branches=N, review_rounds=0)
    asyncio.run(app.ainvoke(st, config={"configurable": {"thread_id": sid}}))
    snap = progress.snapshot(sid)
    assert all(b >= a for a, b in zip(seen, seen[1:]))        # never goes backwards
    assert seen[0] <= 0.1 and 0 < max(seen) < 1
    assert snap["branches_done"] == N and snap["fraction"] >= 0.9 and not snap["finished"]    # UI calls finish()
    progress.finish(sid)
    assert progress.snapshot(sid)["fraction"] == 1.0


def test_failed_branch_still_advances_progress(monkeypatch):
    from orchestrator.graph import app
    sid = str(uuid.uuid4())

    def reader(prompt, n):
        if "aspect 0" in prompt[1]["content"]:
            raise RuntimeError("boom")
        return FakeLLM(N)._default("reader")

    monkeypatch.setattr("llm.generate", FakeLLM(N, {"reader": reader}))
    monkeypatch.setattr("agents.searcher.search_web", make_search(2))
    st = initial_state()
    st["session_id"] = sid
    progress.start(sid, N, 0)
    asyncio.run(app.ainvoke(st, config={"configurable": {"thread_id": sid}}))
    assert progress.snapshot(sid)["branches_done"] == N       # the failed branch still counts as finished
