# eval/ — evaluation layer
#
# Metrics:
#   RAGAS   : faithfulness, answer_relevancy, context_precision
#             context comes from raw search_results / extracted_facts in state
#             judge LLM = Groq (Gemini fallback)
#
#   Custom  : citation_accuracy, source_diversity, hallucination_rate
#
# Ground truth: a small hand-written set of test queries + expected facts
#               defined in eval/fixtures.py — no external benchmark dataset.
