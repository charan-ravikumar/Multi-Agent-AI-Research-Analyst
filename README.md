# Autonomous AI Research Analyst

## Folder Structure

```
ai-research-analyst/
├── agents/          # 6 agents: planner, searcher, reader, synthesizer, critic, writer
├── orchestrator/    # LangGraph state machine + state.py (ResearchState)
├── llm/             # Groq-first / Gemini-fallback thin wrapper (llm/client.py)
├── tools/           # Web search, scraping, academic APIs, PDF parsing
├── memory/          # Redis-only: session state, scratchpads, extracted facts
│                    # NO vector store / Chroma / embeddings
├── data/            # reserved for future local caches / fixtures
├── app/             # Streamlit UI (app/main.py — thin, calls orchestrator only)
├── eval/            # RAGAS (faithfulness, answer_relevancy, context_precision)
│                    # + custom metrics (citation_accuracy, source_diversity,
│                    #   hallucination_rate); ground truth = eval/fixtures.py
├── tests/           # pytest test suite
├── config.py        # pydantic-settings — single source for all env vars
└── .env             # secrets / config (never committed)
```

## Memory model

All working memory is Redis-only (no vector DB):

| What | Redis key pattern | TTL |
|---|---|---|
| Agent scratchpad | `<session_id>:scratchpad:<agent>` | `CACHE_TTL_SECONDS` |
| Extracted facts | `<session_id>:extracted_facts` | `CACHE_TTL_SECONDS` |
| Session state fields | `<session_id>:<field>` | `CACHE_TTL_SECONDS` |

## Evaluation

RAGAS metrics (faithfulness, answer_relevancy, context_precision) run against
`search_results` and `extracted_facts` already in `ResearchState` — no retrieval
step. Custom metrics: citation_accuracy, source_diversity, hallucination_rate.

Ground truth is a small hand-written set of queries + expected facts in
`eval/fixtures.py` — no external benchmark dataset.

## Setup

1. Copy `.env` and fill in your API keys.
2. Install dependencies: `uv pip install -r requirements.txt`
3. Run the app: `streamlit run app/main.py`
