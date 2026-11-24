# eval/ — reproducible evaluation of the research pipeline.
#
#   harness.py   record/replay (search + page text), on-disk LLM cache, metrics, judge, aggregation
#   run_eval.py  CLI:  uv run python -m eval.run_eval --configs no_debate,debate
#   golden_queries.json   the query set
#   fixtures/    recorded search results + fetched page text (committed: same evidence every run)
#   results/     run outputs (JSON) ; RESULTS.md is the human-readable summary
