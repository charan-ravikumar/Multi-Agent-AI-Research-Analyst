from research_analyst.agents.writer import _validate_report

FULL = (
    "## Executive Summary\na\n## Key Findings\nb\n## Detailed Analysis\nc\n"
    "## Contradictions and Conflicting Evidence\nA vs B\n"
    "## Knowledge Gaps and Limitations\nGap one"
)


def test_valid_report_has_no_issues():
    assert _validate_report(FULL, ["A vs B"], ["Gap one"]) == []


def test_missing_header_flagged():
    assert any("key findings" in i for i in _validate_report(FULL.replace("## Key Findings\nb\n", ""), [], []))


def test_empty_disclosure_flagged():
    bad = FULL.replace("Gap one", "No significant gaps were identified.")
    assert any("Knowledge Gaps" in i for i in _validate_report(bad, [], ["Gap one"]))
