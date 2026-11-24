import json
import pytest
from research_analyst.llm.json_parse import parse_json


def test_plain_object():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_fenced():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_literal_newline_inside_string():
    # seen in logs/agent_traces.jsonl as "Invalid control character"
    assert parse_json('{"synthesis": "line one\nline two"}') == {"synthesis": "line one\nline two"}


def test_trailing_text_extra_data():
    # seen in logs/agent_traces.jsonl as "Extra data"
    assert parse_json('{"a": 1}\n\nHope this helps!') == {"a": 1}


def test_leading_prose():
    assert parse_json('Sure, here you go: {"a": [1, 2]}') == {"a": [1, 2]}


def test_array():
    assert parse_json("[1, 2]") == [1, 2]


@pytest.mark.parametrize("raw", [None, "", "   \n"])
def test_empty_raises_value_error(raw):
    with pytest.raises(ValueError):
        parse_json(raw)


def test_garbage_raises_json_error():
    with pytest.raises(json.JSONDecodeError):
        parse_json("not json at all")


def test_truncated_raises():
    with pytest.raises(ValueError):
        parse_json('{"synthesis": "cut off mid-sen')
