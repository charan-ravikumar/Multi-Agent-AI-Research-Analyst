from research_analyst.grounding.verify import (
    Claim, build_evidence_index, extract_claims, highlight, support_for, verify_payload,
)
from research_analyst.models.fact import Fact
from research_analyst.models.search_result import SearchResult

REPORT = """## Executive Summary
Solid-state batteries could roughly double energy density for electric vehicles [1][2].

## Key Findings
- Ceramic electrolytes conduct lithium ions at room temperature, which removes the flammable liquid [1].
- Short line [1].
- Manufacturing costs remain several times higher than for conventional lithium-ion cells today [2].

## Detailed Analysis
No citation in this sentence, so it is not a verifiable claim at all.

## References

[1] A — https://a.example
[2] B — https://b.example
"""


def test_extract_claims_keeps_only_cited_substantial_sentences_before_references():
    claims = extract_claims(REPORT)
    assert [c.refs for c in claims] == [(1, 2), (1,), (2,)]
    assert claims[0].text.startswith("Solid-state batteries could roughly double") and "[" not in claims[0].text
    assert all("Short line" not in c.text and "No citation" not in c.text and "https://a.example" not in c.text for c in claims)
    assert [c.id for c in claims] == [1, 2, 3]


def test_extract_claims_respects_the_cap_and_dedupes():
    many = "\n".join(f"- This is claim number {i} with enough words to count as a sentence [1]." for i in range(60))
    assert len(extract_claims(many, max_claims=10)) == 10
    assert len(extract_claims("The same long sentence is repeated here for sure [1]. The same long sentence is repeated here for sure [1].")) == 1


def test_highlight_marks_shared_terms_and_escapes_everything_else():
    out = highlight('Lithium <script>alert(1)</script> ions & "cells"', {"lithium", "ions"})
    assert out.startswith("<mark>Lithium</mark>") and "<mark>ions</mark>" in out
    assert "<script>" not in out and "&lt;script&gt;" in out and "&amp;" in out
    assert out.count("<mark>") == 2


def test_highlight_cannot_be_broken_by_entity_lookalike_words():
    import html
    import re
    text = "fish &amp; chips amp quot"
    out = highlight(text, {"amp", "quot"})
    assert out.count("<mark>") == 3                                   # 'amp' inside the literal '&amp;' is just a word
    assert html.unescape(re.sub(r"</?mark>", "", out)) == text        # rendering gives back EXACTLY the original text


EVIDENCE = {
    "1": {"title": "A", "url": "https://a.example", "facts": ["Ceramic electrolytes conduct lithium ions at room temperature."],
          "text": "Intro paragraph about the weather and other unrelated matters here.\n\nCeramic electrolytes conduct lithium ions at room "
                  "temperature, which removes the flammable liquid used in conventional cells and improves safety considerably."},
    "2": {"title": "B", "url": "https://b.example", "facts": [], "text": "Pricing discussion: manufacturing costs remain several times "
                                                                   "higher than for conventional lithium-ion cells for now."},
}


def test_support_for_finds_the_matching_passage_and_labels_it():
    c = Claim(1, "Ceramic electrolytes conduct lithium ions at room temperature, which removes the flammable liquid.", (1,))
    [s] = support_for(c, EVIDENCE)
    assert s["label"] == "strong" and s["score"] >= 0.5 and "<mark>Ceramic</mark>" in s["passage_html"]
    assert "weather" not in s["passage_html"] and s["url"] == "https://a.example" and "<mark>" in s["fact_html"]


def test_support_labels_weak_for_unrelated_evidence_and_handles_missing_sources():
    c = Claim(1, "Quantum annealers outperform classical optimisation on logistics benchmarks.", (1, 9))
    s1, s2 = support_for(c, EVIDENCE)
    assert s1["label"] == "weak" and s2["title"] == "(source not found)" and s2["label"] == "weak"


def test_evidence_index_and_payload_roundtrip_through_json():
    import json
    results = [SearchResult(sub_question="q", url="https://a.example", title="A", snippet="short", full_text="long full text " * 5),
               SearchResult(sub_question="q", url="https://b.example", title="B", snippet="only a snippet")]
    facts = [Fact(sub_question="q", content="A fact.", source_url="https://a.example", source_title="A", confidence=0.9)]
    refs = [{"n": 1, "url": "https://a.example", "title": "A"}, {"n": 2, "url": "https://b.example", "title": "B"}]
    idx = build_evidence_index(results, facts, refs)
    assert idx["1"]["text"].startswith("long full text") and idx["2"]["text"] == "only a snippet" and idx["1"]["facts"] == ["A fact."]
    payload = json.loads(json.dumps(verify_payload(REPORT, results, facts, refs)))          # JSON-safe (gallery stores it)
    assert payload["claims"][0]["refs"] == [1, 2] and set(payload["evidence"]) == {"1", "2"}
    assert support_for(Claim(1, "A fact that matters here for the report.", (1,)), payload["evidence"])[0]["ref"] == 1
