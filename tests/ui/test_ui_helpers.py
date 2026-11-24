import json

from research_analyst.ui import trace_tail


def test_trace_tail_reads_only_the_end_of_a_big_file(monkeypatch, tmp_path):
    log = tmp_path / "t.jsonl"
    lines = [json.dumps({"session_id": "old", "step": f"s{i}", "pad": "x" * 200}) for i in range(5000)]
    lines += [json.dumps({"session_id": "mine", "step": f"m{i}"}) for i in range(30)]
    log.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(trace_tail, "LOG_FILE", log)
    out = trace_tail.read_trace_tail("mine", n=10, max_bytes=20_000)
    assert [r["step"] for r in out] == [f"m{i}" for i in range(20, 30)]
    assert trace_tail.read_trace_tail("nobody", max_bytes=20_000) == []


def test_trace_tail_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_tail, "LOG_FILE", tmp_path / "nope.jsonl")
    assert trace_tail.read_trace_tail("x") == []
