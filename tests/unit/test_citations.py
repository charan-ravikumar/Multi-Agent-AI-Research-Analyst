from research_analyst.grounding import citations as cz
from research_analyst.models.citation import Citation
from research_analyst.models.fact import Fact


def fact(url, content, title="T"):
    return Fact(sub_question="q", content=content, source_url=url, source_title=title, confidence=0.9)


A, B = "https://a.example/x", "https://b.example/y"
KA, KB = cz.source_key(A), cz.source_key(B)


def test_fact_markers_convert_and_drop_unknown():
    out, dropped = cz.convert_fact_markers("Claim one. [F1][F2] Claim two. [F9]", {"F1": A, "F2": B})
    assert f"[S:{KA}][S:{KB}]" in out and "[F" not in out and dropped == 1


def test_grouped_ids_and_duplicates():
    out, _ = cz.convert_fact_markers("Claim. [F1, F2]", {"F1": A, "F2": A})
    assert out.count("[S:") == 1


def test_number_citations_first_appearance_and_unknown_removed():
    secs = [f"One. [S:{KB}] two [S:{KA}].", f"Three [S:{KB}] [S:ffffff]."]
    cites = [Citation(url=A, title="A", snippet="s"), Citation(url=B, title="B", snippet="s"), Citation(url=A, title="dup", snippet="s")]
    numbered, refs, unknown = cz.number_citations(secs, cites)
    assert [r["url"] for r in refs] == [B, A] and numbered[0].startswith("One. [1] two [2]") and unknown == 1
    assert "[S:" not in "".join(numbered)


def test_verify_numbered_and_references_are_code_built():
    text, removed = cz.verify_numbered("a [1] b [4] c", {1, 2})
    assert "[4]" not in text and removed == 1
    refs = [{"n": 1, "url": A, "title": "A"}, {"n": 2, "url": B, "title": "B"}]
    out = cz.render_references(refs, cz.cited_numbers(text))
    assert A in out and B not in out
    assert "No claim" in cz.render_references(refs, set())


def test_analyze_flags_uncited_and_weak_support():
    facts = [fact(A, "Machine learning shortens drug screening cycles"), fact(B, "Lisbon weather was sunny on Tuesday")]
    secs = [
        f"Machine learning shortens early drug screening cycles a lot. [S:{KA}] "
        f"Quantum computers will certainly replace every laboratory by next year. "
        f"Machine learning shortens early drug screening cycles a lot. [S:{KB}]"
    ]
    s = cz.analyze(secs, facts)
    assert s["claim_sentences"] == 3 and s["cited"] == 2 and s["uncited"] == 1 and s["weakly_supported"] == 1
    assert any(p.startswith("UNCITED") for p in s["problems"]) and any(p.startswith("WEAK") for p in s["problems"])


def test_build_evidence_is_keyed_and_capped():
    ev = cz.build_evidence([fact(A, "alpha fact"), fact(B, "beta fact")], max_chars=10_000)
    assert f"[S:{KA}]" in ev and "alpha fact" in ev
    assert "truncated" in cz.build_evidence([fact(A, "x" * 500), fact(B, "y" * 500)], max_chars=100)
