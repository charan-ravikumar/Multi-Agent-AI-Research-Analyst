"""Headless Streamlit tests (AppTest): runs the real app/main.py script, no network."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.about import SAMPLE_QUESTIONS  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    return at


def test_start_page_has_about_tab_and_example_questions():
    at = _app().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["🔬 Research", "ℹ️ About this app"]
    about = " ".join(m.value for m in at.tabs[1].markdown)
    assert "What is this?" in about and "How it works" in about and "Built with" in about
    samples = [b for b in at.button if b.key and b.key.startswith("sample_")]
    assert len(samples) == len(SAMPLE_QUESTIONS) == 5


def test_clicking_an_example_fills_the_question_box():
    at = _app().run()
    next(b for b in at.button if b.key == "sample_1").click()
    at.run()
    assert at.text_input(key="query_input").value == SAMPLE_QUESTIONS[1]
    assert not at.exception


def _done_state(open_questions=True):
    items = [{"sub_question": "How is X used?", "kind": "coverage", "text": 'No search results were found for sub-question: "How is X used?"'}]
    if open_questions:
        items += [{"sub_question": "How is X used?", "kind": "open_question", "text": "What is the long-term cost of X?"},
                  {"sub_question": "Who regulates X?", "kind": "open_question", "text": "How do EU and US rules on X differ?"}]
    return {"final_report": "## Executive Summary\nHello [1].\n\n## References\n\n[1] T — https://example.com", "query": "Topic X",
            "gap_items": items, "contradictions": ["Source A and B disagree on X"], "citations": [], "routing_history": [],
            "usage_summary": {"calls": 3, "total_tokens": 1234, "by_agent": {}, "by_model": {}}}


def _done_app(state):
    at = _app()
    at.session_state["phase"] = "done"
    at.session_state["query"] = "Topic X"
    at.session_state["session_id"] = "sid"
    at.session_state["final_state"] = state
    return at.run()


def test_results_page_separates_coverage_warnings_from_follow_up_research():
    at = _done_app(_done_state())
    assert not at.exception
    heads = [s.value for s in at.subheader]
    assert any("Coverage warnings" in h for h in heads) and any("follow-up research" in h for h in heads)
    assert any("No search results were found" in w.value for w in at.warning)
    follow = [b for b in at.button if b.key and b.key.startswith("followup_")]
    assert len(follow) == 2
    assert any("Contradictions between sources" in e.label for e in at.expander)


def test_follow_up_button_restarts_with_the_question_prefilled():
    at = _done_app(_done_state())
    next(b for b in at.button if b.key == "followup_0").click()
    at.run()
    assert at.session_state["phase"] == "idle"
    assert "What is the long-term cost of X?" in at.text_input(key="query_input").value
    assert not at.exception


def test_legacy_states_without_gap_items_still_render():
    st_ = _done_state(open_questions=False)
    st_["gap_items"] = []
    st_["unresolved_gaps"] = ["Some old style gap"]
    at = _done_app(st_)
    assert not at.exception and any("Gaps noted" in e.label for e in at.expander)
