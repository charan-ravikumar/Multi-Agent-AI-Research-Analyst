import memory.redis_store as rs


def test_related_reports_by_token_overlap():
    rs.save_report_record(session_id="s1", query="machine learning drug discovery pipelines",
                          final_report="# T\nSummary text here.", sub_questions=["How does ML speed drug discovery?"])
    rs.save_report_record(session_id="s2", query="medieval castle architecture",
                          final_report="x", sub_questions=["What were moats for?"])
    hits = rs.find_related_reports("drug discovery with machine learning")
    assert len(hits) == 1 and hits[0]["session_id"] == "s1"
    assert {"drug", "discovery", "machine", "learning"} <= set(hits[0]["_overlap_tokens"])


def test_no_overlap_returns_empty():
    rs.save_report_record(session_id="s1", query="quantum computing", final_report="x", sub_questions=[])
    assert rs.find_related_reports("gardening tomatoes") == []


def test_index_is_capped():
    for i in range(rs._REPORTS_MAX_SIZE + 5):
        rs.save_report_record(session_id=str(i), query=f"topic {i}", final_report="x", sub_questions=[])
    assert rs.get_client().zcard(rs._REPORTS_INDEX_KEY) == rs._REPORTS_MAX_SIZE
