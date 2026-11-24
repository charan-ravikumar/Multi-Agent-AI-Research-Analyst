import json

from streamlit.testing.v1 import AppTest

from research_analyst.ui import gallery
from tests.ui.test_ui_app import APP, _verify_state


def _entry():
    from research_analyst.grounding.verify import verify_payload
    s = _verify_state()
    v = verify_payload(s["final_report"], s["search_results"], s["extracted_facts"], s["references"])
    return {"id": "topic-x", "question": "What about topic X?", "generated_at": "2026-11-24", "models": {"strong": "m-120b"},
            "review_loop": False, "seconds": 84.0, "usage": {"total_tokens": 19000, "calls": 17, "estimated_cost_usd": None},
            "report": s["final_report"], "references": s["references"], "citation_stats": {"claim_sentences": 2, "cited": 2, "uncited": 0,
            "weakly_supported": 0, "sources_cited": 2, "sources_consulted": 2, "invalid_markers_removed": 0},
            "coverage": ["One coverage warning."], "open_questions": [{"text": "What about cost over time?", "sub_question": "How costly?"}],
            "contradictions": [], "verify": v}


def test_load_gallery_skips_malformed_files(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(_entry()))
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "empty.json").write_text("{}")
    assert [e["id"] for e in gallery.load_gallery(tmp_path)] == ["topic-x"]
    assert gallery.load_gallery(tmp_path / "missing") == []


def test_slugify():
    assert gallery.slugify("What is the *current* state of CRISPR?") == "what-is-the-current-state-of-crispr"


def test_entry_from_state_is_json_safe_and_caps_evidence():
    s = _verify_state()
    s["search_results"][0] = s["search_results"][0].model_copy(update={"full_text": "word " * 1000})
    s.update({"usage_summary": {"total_tokens": 5, "calls": 1}, "citation_stats": {"cited": 1}, "contradictions": ["a vs b"],
              "gap_items": [{"sub_question": "q", "kind": "open_question", "text": "A follow-up?"},
                            {"sub_question": "q", "kind": "coverage", "text": "A warning"}]})
    e = gallery.entry_from_state(s, "Q?", 12.34, {"fast": "f", "strong": "s"})
    json.dumps(e)                                                        # must serialise
    assert e["seconds"] == 12.3 and e["coverage"] == ["A warning"] and e["open_questions"][0]["text"] == "A follow-up?"
    assert all(len(v["text"]) <= 900 for v in e["verify"]["evidence"].values())
    assert e["verify"]["claims"] and e["usage"]["total_tokens"] == 5


def test_gallery_tab_renders_report_metrics_and_a_working_verify_panel(monkeypatch, tmp_path):
    (tmp_path / "topic-x.json").write_text(json.dumps(_entry()))
    monkeypatch.setattr(gallery, "GALLERY_DIR", tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    tab = at.tabs[1]
    text = " ".join(m.value for m in tab.markdown)
    assert "Sample reports" in text and "Ceramic electrolytes" in text
    assert {m.label: m.value for m in at.metric}["Tokens"] == "19,000"
    assert any("One coverage warning." in w.value for w in tab.warning)
    claims = [b for b in at.button if b.key and b.key.startswith("gallery_topic-x_claim_")]
    assert len(claims) == 2
    claims[1].click()
    at.run()
    assert not at.exception and "<mark>" in " ".join(m.value for m in at.tabs[1].markdown)


def test_empty_gallery_is_a_friendly_message(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "GALLERY_DIR", tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("No sample reports" in i.value for i in at.tabs[1].info)
