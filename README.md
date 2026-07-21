# Autonomous AI Research Analyst

A multi-agent AI research pipeline that turns a natural-language topic into a structured, comprehensive research report. The system uses LangGraph for orchestration, a set of specialized agents for planning, searching, reading, synthesizing, critiquing, and writing, with a Redis-backed memory layer for cross-run context.

---

## 🎯 Key Features

- **Multi-Agent Architecture**: Uses six specialized agents with distinct roles (Strategy, Discovery, Comprehension, Analysis, Quality Control, and Writing).
- **Stateful Graph Orchestration**: Managed by a LangGraph state machine supporting parallel fan-out and fan-in execution paths.
- **Reflection & Debate Loop**: A Critic agent evaluates drafts for quality, triggering iterative refinements (up to a safety cap of 3 rounds to control API costs).
- **Proactive Concurrency & Rate Control**: Designed for free-tier constraints (Groq as primary, Gemini as fallback) with managed request limits and retry/backoff wrappers.
- **Cross-Run memory**: Uses Redis for session scratchpads, report lookup, and persistence of metadata.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────┐
│           STREAMLIT UI (Single App)         │
│     Shows live agent progress + report       │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│      ORCHESTRATOR (LangGraph State Machine)   │
│                                               │
│   ┌─────────┐   ┌──────────┐   ┌──────────┐   │
│   │ Planner │──▶│ Searcher │──▶│  Reader  │   │
│   └─────────┘   └──────────┘   └────┬─────┘   │
│        ▲                             │         │
│        │        ┌──────────┐   ┌────▼──────┐  │
│        └────────│  Critic  │◀──│Synthesizer│  │
│                 └──────────┘   └────┬──────┘  │
│                                     │          │
│                               ┌─────▼────┐    │
│                               │  Writer  │    │
│                               └──────────┘    │
└─────────────┬─────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌──────┐          ┌──────────┐
│Redis │          │Local logs│
│(STM) │          │/Langfuse │
└──────┘          └──────────┘
```

### The 6 Agents

| Agent | Role | Key Responsibilities |
|---|---|---|
| 🧠 **Planner** | Strategy | Decomposes the query into sub-questions, creates the research plan, and decides depth. |
| 🔍 **Searcher** | Discovery | Queries free search APIs (DuckDuckGo, Scholar via `scholarly`) and academic APIs (arXiv), scrapes with Playwright. |
| 📖 **Reader** | Comprehension | Chunks content and extracts key facts. |
| 🧪 **Synthesizer** | Analysis | Cross-references sources, merges claims, and flags contradictions or gaps. |
| 🕵️ **Critic** | Quality Control | Fact-checks claims, scores relevance, and raises specific objections to improve the draft. |
| ✍️ **Writer** | Output | Generates the final structured Markdown report with references, citations, and summaries. |

---

## 📁 Repository Layout

```text
agents/             Agent classes: planner, searcher, reader, synthesizer, critic, writer
app/                Streamlit UI entry point
core/               Shared logging utilities
config.py           Central settings and environment configuration
llm/                LLM provider wrappers and fallback logic (Groq / Gemini)
memory/             Redis-backed persistence helpers
models/             Pydantic and TypedDict models for state and outputs
orchestrator/       LangGraph graph definitions and state wiring
tools/              Demo runner, evaluation harness, replay utilities, web search
tests/              Automated and manual test scripts
```

---

## 🛠️ Tech Stack

- **Orchestration**: LangGraph, LangChain
- **LLMs**: Groq (primary, free tier), Google Gemini (fallback, free tier)
- **Memory**: Redis (short-term state, report index, and session metadata)
- **Tools**: `duckduckgo-search`, `scholarly`, Playwright, arXiv, PyMuPDF, `unstructured`
- **Backend & UI**: Python, Pydantic, Streamlit
- **Observability**: Local structured logging, Langfuse tracing

---

## 🚀 Setup & Installation

### 1. Prerequisites

- **Python 3.10+** (target version: 3.12)
- **[uv](https://github.com/astral-sh/uv)** (recommended for dependency and environment management)
- **Redis** (optional but recommended for memory features; degrades gracefully if unavailable)
- **API Keys** for Groq and Google Gemini

### 2. Install Dependencies

Use `uv` to automatically initialize the virtual environment and synchronize dependencies:
```bash
uv sync
```

### 3. Environment Configuration

Create a `.env` file in the root of the project:
```env
# Primary LLM API Key
GROQ_API_KEY=your_groq_api_key_here

# Fallback LLM API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Redis URL (e.g. local)
REDIS_URL=redis://localhost:6379/0
```

---

## 💻 Running the Application

### Start the Streamlit UI
To launch the interactive web application, run:
```bash
uv run streamlit run app/main.py
```

### Running the Demo without UI
To run a test research query via the command-line demo:
```bash
uv run python tools/run_demo.py
```

### Testing
To run the verification test suite:
```bash
uv run python tests/test_v1_send_isolation.py
uv run python tests/test_v2_retry_parallel.py
uv run python tests/test_v3_debate_cap.py
```
