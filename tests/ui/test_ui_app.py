"""Headless Streamlit tests (AppTest): runs the real app/main.py script, no network."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from research_analyst.ui.about import SAMPLE_QUESTIONS  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "..", "app", "main.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    return at


def test_start_page_has_about_tab_and_example_questions():
    at = _app().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Research", "Sample reports", "Monitor", "About"]
    about = " ".join(m.value for m in at.tabs[3].markdown)
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


# ── verify panel, run summary, cancel, preflight ──────────────────────────────

def _verify_state():
    from research_analyst.models.fact import Fact
    from research_analyst.models.search_result import SearchResult
    report = ("## Executive Summary\nCeramic electrolytes conduct lithium ions at room temperature, removing flammable liquids [1].\n\n"
              "## Key Findings\n- Manufacturing costs remain several times higher than for conventional lithium-ion cells today [2].\n\n"
              "## References\n\n[1] A — https://a.example\n[2] B — https://b.example")
    return {
        "final_report": report, "query": "Topic X", "gap_items": [], "contradictions": [], "citations": [], "routing_history": [],
        "references": [{"n": 1, "url": "https://a.example", "title": "A"}, {"n": 2, "url": "https://b.example", "title": "B"}],
        "search_results": [
            SearchResult(sub_question="q", url="https://a.example", title="A", snippet="s",
                         full_text="Ceramic electrolytes conduct lithium ions at room temperature. <script>alert(1)</script>"),
            SearchResult(sub_question="q", url="https://b.example", title="B", snippet="Manufacturing costs remain several times higher than conventional cells."),
        ],
        "extracted_facts": [Fact(sub_question="q", content="Ceramic electrolytes conduct lithium ions.", source_url="https://a.example",
                                 source_title="A", confidence=0.9)],
        "usage_summary": {"calls": 7, "total_tokens": 12345, "by_agent": {}, "by_model": {}, "estimated_cost_usd": None, "budget": 0},
    }


def test_results_page_lists_claims_and_shows_highlighted_evidence_for_the_selected_one():
    at = _done_app(_verify_state())
    assert not at.exception
    claim_buttons = [b for b in at.button if b.key and b.key.startswith("live_claim_")]
    assert len(claim_buttons) == 2
    first_html = " ".join(m.value for m in at.markdown)
    assert "<mark>Ceramic</mark>" in first_html and "<script>" not in first_html          # highlighted AND escaped
    claim_buttons[1].click()
    at.run()
    html_after = " ".join(m.value for m in at.markdown)
    assert "<mark>Manufacturing</mark>" in html_after and "Ceramic electrolytes conduct" not in html_after.split("Claim 2")[1]
    assert not at.exception


def test_results_page_shows_time_tokens_calls_and_free_tier_cost():
    at = _done_app(_verify_state())
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Tokens"] == "12,345" and metrics["LLM calls"] == "7" and metrics["Cost"].startswith("$0")


def test_results_page_shows_an_estimate_when_prices_are_configured():
    s = _verify_state()
    s["usage_summary"]["estimated_cost_usd"] = 0.0421
    at = _done_app(s)
    assert {m.label: m.value for m in at.metric}["Estimated cost"] == "$0.0421"


def test_cancelled_run_shows_a_stopped_screen_and_can_restart():
    from research_analyst.observability.usage import tracker
    tracker.record("sid-c", "reader", {"model": "m", "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    at = _app()
    at.session_state["phase"] = "cancelled"
    at.session_state["query"] = "Topic X"
    at.session_state["session_id"] = "sid-c"
    at.run()
    assert not at.exception and any("Research stopped" in w.value for w in at.warning)
    assert any("120 tokens" in c.value for c in at.caption)
    next(b for b in at.button if b.label == "Start over").click()
    at.run()
    assert at.session_state["phase"] == "idle"


def test_a_cancelled_thread_routes_the_researching_page_to_the_stopped_screen():
    import threading
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    at = _app()
    at.session_state["phase"] = "researching"
    at.session_state["query"] = "Topic X"
    at.session_state["session_id"] = "sid-r"
    at.session_state["bg_thread"] = t
    at.session_state["bg_result"] = {"done": True, "cancelled": True}
    at.run()
    assert not at.exception and at.session_state["phase"] == "cancelled"


def test_start_page_blocks_when_a_configured_model_is_unavailable(monkeypatch):
    from research_analyst.llm import preflight
    from research_analyst.llm.preflight import ModelCheck
    from research_analyst.config import settings
    monkeypatch.setattr(settings, "gemini_api_key", "x")
    monkeypatch.setattr(preflight, "check_models", lambda force=False: ModelCheck(
        verified=True, available=("gemini-3.5-flash-lite",), missing=("gemini-3.8-flash",), suggestions=("gemini-3.5-flash-lite",)))
    at = _app().run()
    assert any("gemini-3.8-flash" in e.value and "no fallback" in e.value for e in at.error)
    start = next(b for b in at.button if b.label.startswith("Start research"))
    assert start.disabled


def test_start_page_without_a_key_blocks_starting(monkeypatch):
    from research_analyst.config import settings
    monkeypatch.setattr(settings, "gemini_api_key", "")
    at = _app().run()
    assert any("No LLM API key" in e.value and "GEMINI_API_KEY" in e.value for e in at.error)
    assert next(b for b in at.button if b.label.startswith("Start research")).disabled
