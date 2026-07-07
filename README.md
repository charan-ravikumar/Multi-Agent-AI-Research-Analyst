# Autonomous AI Research Analyst

This repository contains an agentic research pipeline that turns a natural-language topic into a structured research report. The system uses LangGraph for orchestration, a set of specialized agents for planning, search, reading, synthesis, critique, and writing, and a Redis-backed memory layer for cross-run context.


## What this project does

Given a research question, the pipeline:

1. Plans a set of sub-questions.
2. Lets a human approve or edit the plan.
3. Runs parallel research branches for each sub-question.
4. Extracts facts from search results.
5. Synthesizes draft sections and flags contradictions or gaps.
6. Runs a bounded critique-and-revision loop.
7. Writes a Markdown report that explicitly discloses unresolved concerns.
8. Persists a compact record for future retrieval and related-report lookup.

The design is intentionally optimized for a constrained, free-tier environment: DuckDuckGo for search, Groq as the primary LLM provider, Gemini as a fallback, Redis for memory, and local JSON-lines logging.

## Core architecture

The system is organized around a shared typed state object and a LangGraph workflow.

### Main components

- Planning: the planner agent decomposes a topic into sub-questions.
- Search: the searcher agent gathers results for each sub-question.
- Reading: the reader agent extracts factual claims from search snippets.
- Synthesis: the synthesizer agent assembles draft content, contradictions, and knowledge gaps.
- Critique: the critic agent raises specific objections to improve the draft.
- Writing: the writer agent produces the final Markdown report.
- Orchestration: the graph in the orchestrator package coordinates the full workflow, including fan-out, fan-in, debate rounds, checkpoints, and memory persistence.

### Key implementation choices

- LangGraph-based state machine with parallel fan-out and fan-in.
- Shared state schema defined in the models package.
- Best-effort Redis memory layer with graceful degradation if Redis is unavailable.
- Structured logging for traceability and debugging.
- Human-in-the-loop plan approval to allow editing before research begins.

## Repository layout

```text
agents/             Agent classes: planner, searcher, reader, synthesizer, critic, writer
app/                Streamlit UI entry point
core/               Shared logging utilities
config.py           Central settings and environment configuration
llm/                LLM provider wrappers and fallback logic
memory/             Redis-backed persistence helpers
models/             Pydantic and TypedDict models for state and outputs
orchestrator/       LangGraph graph definitions and state wiring
tools/              Demo runner, evaluation harness, replay utilities, web search
tests/              Automated and manual test scripts
files (4)/          Detailed project documentation and build notes
```

## Setup

### Prerequisites

- Python 3.9+ (the project targets modern Python versions)
- Access to the required API keys for Groq and Gemini
- Redis is optional but recommended for memory features

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Environment variables

Create a .env file in the project root with values such as:

```env
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

The project uses pydantic-settings, so environment values are read centrally from config.py.

## Running the project

### Streamlit UI

```bash
streamlit run app/main.py
```

### Demo run without the UI

```bash
python tools/run_demo.py
```

### Trace and evaluate runs

```bash
python tools/trace_summary.py --last
python tools/eval_harness.py
```

## Testing

The repository includes both automated and manual test scripts.

- Automated tests cover graph behavior, fan-out/fan-in behavior, debate caps, checkpoints, and memory persistence.
- Manual tests exercise live LLM and web-search interactions.

Example:

```bash
python tests/test_v1_send_isolation.py
python tests/test_v2_retry_parallel.py
python tests/test_v3_debate_cap.py
```

## Memory and persistence

The system uses Redis for:

- cross-run report lookup
- session scratchpads
- persistence of compact report metadata

If Redis is unavailable, the pipeline is designed to degrade gracefully and continue without those features rather than failing outright.

## Known limitations

The current implementation is functional, but it has a few important limitations:

- full-page scraping is not yet implemented, so the reader works from snippet text rather than full page content
- the Gemini fallback depends on working provider SDK support in the environment
- the debate loop is intentionally capped to keep cost under control
- memory matching is keyword-based rather than semantic vector search

