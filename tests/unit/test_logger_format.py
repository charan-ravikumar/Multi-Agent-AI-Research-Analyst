import json
import logging

from research_analyst.config import settings
from research_analyst.observability import logger


def _stream_formatter(monkeypatch, tmp_path, env, fmt):
    monkeypatch.setattr(settings, "environment", env)
    monkeypatch.setattr(settings, "log_format", fmt)
    monkeypatch.setattr(settings, "log_file", str(tmp_path / "t.jsonl"))
    handlers = logger._build_handlers()
    for h in handlers:
        h.close()
    return handlers[0].formatter


def _render(formatter):
    rec = logging.LogRecord("agents.x", logging.INFO, __file__, 1, "hello", None, None)
    rec.step = "s"
    rec.session_id = "sid"
    return formatter.format(rec)


def test_text_in_development_by_default(monkeypatch, tmp_path):
    out = _render(_stream_formatter(monkeypatch, tmp_path, "development", "text"))
    assert "hello" in out and not out.startswith("{")


def test_json_when_requested_even_in_development(monkeypatch, tmp_path):
    d = json.loads(_render(_stream_formatter(monkeypatch, tmp_path, "development", "json")))
    assert d["message"] == "hello" and d["step"] == "s" and d["session_id"] == "sid"


def test_json_in_production(monkeypatch, tmp_path):
    assert json.loads(_render(_stream_formatter(monkeypatch, tmp_path, "production", "text")))["message"] == "hello"
