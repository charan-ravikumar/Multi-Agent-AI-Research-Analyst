# Evaluation results

- queries: **3** (each run once; paired comparison `debate` vs `no_debate`)
- generator: Groq `openai/gpt-oss-20b / openai/gpt-oss-120b`; judge: `qwen/qwen3.8-27b`; judge model family differs from the generator in 6/6 runs
- evidence: recorded search + page fixtures (identical across configs)

| metric | no_debate | debate |
|---|---|---|
| fact_fidelity | 1.0 | 1.0 |
| cited_ratio | 0.79 | 0.913 |
| judge_faithfulness | 0.75 | 0.779 |
| judge_completeness | 4.333 | 3.667 |
| judge_clarity | 5 | 5 |
| uncited | 4 | 2 |
| weakly_supported | 0 | 0 |
| judge_unsupported_claims | 0.667 | 0.667 |
| report_words | 1146.333 | 1169.333 |
| tokens_nominal | 15396 | 26920.667 |
| llm_calls_nominal | 11.667 | 14.667 |
| gaps | 16 | 16 |
| debate_rounds | 0 | 2 |
| review_notes | 0 | 0 |
| sources_cited | 9.667 | 7 |

## Paired difference: `debate` − `no_debate`

| metric | mean Δ | other better | other worse | ties | n |
|---|---|---|---|---|---|
| fact_fidelity (↑ better) | 0.0 | 0 | 0 | 3 | 3 |
| cited_ratio (↑ better) | 0.123 | 2 | 1 | 0 | 3 |
| judge_faithfulness (↑ better) | 0.029 | 1 | 1 | 1 | 3 |
| judge_completeness (↑ better) | -0.667 | 0 | 2 | 1 | 3 |
| judge_clarity (↑ better) | 0 | 0 | 0 | 3 | 3 |
| uncited (↓ better) | -2 | 2 | 1 | 0 | 3 |
| weakly_supported (↓ better) | 0 | 0 | 0 | 3 | 3 |
| judge_unsupported_claims (↓ better) | 0 | 0 | 0 | 3 | 3 |

> Small n, single run per query, LLM-extracted evidence: read these as directional, not significant.
