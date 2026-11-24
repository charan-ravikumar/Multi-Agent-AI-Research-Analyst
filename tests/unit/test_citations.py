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


# ── citation bracket normalisation (a real run lost every citation to 【1】) ────────

def test_fullwidth_and_browsing_style_brackets_become_ascii():
    n = cz.normalize_citation_brackets
    assert n("Fact one【1】 and two【2】【3】.") == "Fact one[1] and two[2][3]."
    assert n("Fact〔4〕 and ［5］.") == "Fact[4] and [5]."
    assert n("Fact【1†source】 and 【3†L5-L9】.") == "Fact[1] and [3]."


def test_lists_and_ranges_expand_only_for_plausible_reference_numbers():
    n = cz.normalize_citation_brackets
    assert n("Claim [1, 2] and [3;4].", max_ref=5) == "Claim [1][2] and [3][4]."
    assert n("Claim [2-4].", max_ref=5) == "Claim [2][3][4]."
    assert n("Years [2020, 2021] stay.", max_ref=5) == "Years [2020, 2021] stay."          # not references
    assert n("Huge range [1-90].", max_ref=100) == "Huge range [1-90]."                    # not a citation range


def test_internal_markers_are_never_touched():
    n = cz.normalize_citation_brackets
    s = f"Claim [S:{KA}] and [F1, F2] and plain [text]."
    assert n(s) == s


def test_report_citation_stats_measure_the_final_text():
    body = ("## Key Findings\n- Ceramic electrolytes conduct lithium ions at room temperature [1].\n"
            "- An uncited but long enough sentence about manufacturing costs today.\n## Detailed Analysis\nShort [2].\n")
    s = cz.report_citation_stats(body)
    assert s == {"report_claim_sentences": 2, "report_cited": 1, "report_cited_ratio": 0.5}


def test_report_citation_stats_ignore_the_disclosure_sections():
    body = ("## Executive Summary\nSolid-state cells promise much higher energy density for vehicles [1].\n"
            "## Key Findings\n- Ceramic electrolytes conduct lithium ions at room temperature today [1].\n"
            "## Detailed Analysis\nAn uncited but sufficiently long sentence about costs and scaling.\n"
            "## Contradictions and Conflicting Evidence\nSources disagree about the commercial timeline for 2027.\n"
            "## Knowledge Gaps and Limitations\n- Open question: how will recycling work at scale for these cells?\n")
    assert cz.report_citation_stats(body) == {"report_claim_sentences": 3, "report_cited": 2, "report_cited_ratio": 0.667}
