# AI Research Analyst

Ask a question, get a **sourced report**: the app plans sub-questions (you can edit them), searches the web and
arXiv, reads the best passages, and writes a report in which every claim carries a numbered citation — with a
reference list built by code from the sources actually cited, and a panel that shows the exact passage behind
each claim.

It is a staged pipeline of LLM calls on LangGraph (parallel per sub-question, a human approval step, optional
review loop) — not an autonomous agent swarm. Read [Limitations](#limitations) before trusting any output.

**More:** [architecture and failure modes](docs/ARCHITECTURE.md) ·
[what broke when I ran it on free-tier APIs](docs/STORY.md)

## Try it

- **Sample reports tab** — five real, pre-generated reports with their cost and grounding stats and the
  claim-verification panel. Uses no API quota.
- **Research tab** — type a question or click an example. Typical run: about 2–3 minutes, ~22k tokens (measured below).
- **Locally** — see [Run it](#run-it).

## What it does

| Stage | What happens |
|---|---|
| Plan | An LLM splits the question into 4–5 sub-questions; you review and edit them |
| Search | Optional **ReAct mode** (`SEARCH_MODE=react`): the model chooses searches / page reads step by step (max `REACT_MAX_STEPS`) from what it has seen, then the same ranking runs. Default **fixed mode**, per sub-question, in parallel: two phrasings (verbatim + a rewritten keyword query), DuckDuckGo plus relevant **arXiv** papers, rank fusion, a source-quality prior and a per-domain preference, then page text for the top results (SSRF-guarded fetch, best passages by BM25) and **PDF full text** for the top arXiv papers |
| Read | Facts are extracted in batches of 3 sources and reading **stops early** when enough distinct facts exist or a further batch adds nothing new |
| Write | Sections cite facts inline; the Writer is told to use only the draft's facts. Citations are numbered **in code**, invented markers are removed, and `[n]` brackets are normalised (a real run once lost every citation to `【1】`) |
| Check | An **LLM judge** (a different model family) rates every cited claim supported / partial / unsupported against the passages it cites; **click any claim to see the verdict, the reason and the source passage**. Citation-marker counts are mechanical; word overlap is only the fallback if the judge is unavailable |
| Review (opt-in) | Critic ⇄ Reviser loop, off by default (see [Evaluation](#evaluation)) |

Around the pipeline: progress bar with live activity, **Stop** button, token/cost/time per run, coverage warnings
(problems the system detected) separate from **suggested follow-up research** (one click starts a new run),
Langfuse tracing with quality scores, **versioned prompts (Langfuse Prompt Management; the repo text is the fallback)**, a token budget, and a start-page check that a configured model still exists.

## Cost and speed (measured)

Five sample reports, one run each, **on the earlier Groq setup** (free-tier `openai/gpt-oss-20b` for reading, `-120b` for the
rest) — these numbers predate the move to Gemini-only and have not been re-measured; default settings:

| Report | Time | Tokens | LLM calls | Sources cited | Claims cited (final report) | Coverage warnings |
|---|---|---|---|---|---|---|
| LLMs in healthcare | 2:35 | 24,312 | 22 | 20 (7 arXiv) | 40/44 (91%) | 0 |
| Solid-state batteries and EVs | 1:28 | 20,769 | 21 | 12 (2 arXiv) | 27/30 (90%) | 0 |
| CRISPR gene therapies | 2:11 | 20,968 | 22 | 12 (0 arXiv) | 23/28 (82%) | 0 |
| Remote work and productivity | 2:22 | 21,547 | 21 | 13 (0 arXiv) | 29/32 (91%) | 0 |
| Quantum error correction | 2:16 | 24,226 | 22 | 17 (8 arXiv) | 36/41 (88%) | 0 |
| **Median / mean** | **2:16** (median) | **21,547** (median) | **22** (median) | | **88%** (mean) | |

Cost is **$0 on free tiers**; set `LLM_PRICE_PER_1M_TOKENS` to see a paid-rate estimate in the UI. Time is
dominated by search and by the 8,000 tokens/minute free-tier limit, not by compute.
"Claims cited" counts sentences in the Executive Summary, Key Findings and Detailed Analysis that carry a citation
marker in the **final report** (the disclosure sections are legitimately uncited). It measures coverage of
citations, **not** that the claims are true.

## Reliability and safety

Each item below is covered by offline tests; where a live run exposed the problem it is in
[docs/STORY.md](docs/STORY.md).

- A failing sub-question does not abort the run: it becomes a coverage warning and the report is marked partial.
  The run only fails if every branch fails.
- **Gemini only, no fallback provider.** One retry policy: jittered backoff honouring `Retry-After`, a per-model circuit
  breaker that fails fast, request timeouts, and free-tier limits enforced in code per model (requests/minute and
  tokens/minute are waited for; the requests/day budget is counted, persisted and raises a clear error instead of
  burning retries). A fast model (`gemini-3.5-flash-lite`) serves most agents; the stronger `gemini-3.8-flash` (only 20
  requests/day on the free tier) is used only for the agents in `GEMINI_STRONG_AGENTS` (default: the Writer).
- **Native function calling for the ReAct search** (tools declared to the API, mode `ANY`: the model can only answer with
  a call to a declared tool, arguments checked against a JSON schema) and **schema-constrained structured outputs**
  (Pydantic `response_schema`) for the Planner, Reader, Synthesizer, Critic, query rewrite and grounding judge. The API
  guarantees the SHAPE; each agent still validates meaning, retries once with its own error, and a truncated answer
  is an error, never a score. The Writer's Markdown report is plain text by design.
- Untrusted web text is stripped of control characters, forged markers and delimiter tags and wrapped in data
  tags; the final Markdown cannot render images, raw HTML or non-http(s) links. Fetching refuses non-public
  addresses (including via redirects); arXiv XML is parsed with `defusedxml`.
- A `TOKEN_BUDGET_PER_RUN` hard stop degrades gracefully (the Writer is exempt so a report is still produced).

## Evaluation

`uv run python -m research_analyst.evaluation.run_eval --queries 3 --configs no_debate,debate` records search
results and caches LLM calls so both configurations see identical evidence. Results below were produced on the earlier Groq setup (generator Groq `gpt-oss`;
judge `qwen/qwen3.8-27b`, a different model family); the judge now defaults to the same Gemini family as the generator, so
self-preference bias is possible. It scored up to 10 cited claims per report against the facts
extracted from their cited sources. Raw results: `src/research_analyst/evaluation/results/`.

| 3 queries, 1 run each | no review loop | review loop (2 rounds) |
|---|---|---|
| claim sentences that carry a citation (draft) | 79% | 91% |
| judge faithfulness (supported = 1, partial = ½) | 0.75 | 0.78 |
| judge completeness (1–5) | 4.3 | 3.7 |
| tokens per run | 15.4k | 26.9k |
| loop ended at the round cap | – | 3 of 3 |

The review loop mostly adds citation markers; faithfulness is flat, completeness drops, and it costs ~75% more
tokens, so it is **off by default** (a checkbox on the start page, or `DEBATE_MAX_ROUNDS=2`). With n=3 none of this
is statistically meaningful, and the judge only sees extracted facts — it measures "supported by the gathered
evidence", not truth.

**Not yet measured:** the review-loop numbers above predate query rewriting, source ranking, arXiv full text,
early-stop reading and the Writer prompt change, so the end-to-end effect of those features on answer quality is
unmeasured. One thing I did observe: the first five sample reports had only **66%** of claim-section sentences
cited in the final text (the draft was ~81%) because the Writer was told to "expand … exhaustive, elaborate" and
padded the analysis with unsourced sentences. After telling it to use only the draft's facts and keep `[n]`
markers, the same five questions came out at **88%** (table above). That before/after is real but uncontrolled —
search results differ between runs, so it is evidence of a direction, not a measured effect size. Queries 4–5 of
the golden set are defined but unrun (provider daily quotas). Scraped page text is not committed (re-fetched on
first run).

## Limitations

- **Evidence is bounded by DuckDuckGo/arXiv and what `trafilatura`/`pypdf` can extract.** No paywalled or
  JS-rendered pages. Source quality is a hand-written domain/recency prior, not a trained model.
- **The grounding judge is an LLM reading the gathered passages, not the live web.** A "supported" verdict means the
  passages back the claim, not that the claim is true; the judge can be wrong and has **not yet been validated
  against human labels** (nor were the ReAct mode's effect on quality or cost measured). Facts are LLM-extracted. Search results vary between runs, so the same question can yield different
  conclusions.
- **Free-tier limits bite, and there is no fallback:** `gemini-3.8-flash` allows 5 requests/minute and 20/day,
  `gemini-3.5-flash-lite` 15/minute and 500/day (about 25 calls per run, so roughly 20 runs/day; more in ReAct mode). If a
  model is overloaded (503/504, seen on `gemini-3.8-flash`) retries are exhausted and the run fails — set
  `GEMINI_STRONG_AGENTS=` (empty) to run everything on the fast model. arXiv rate-limits by IP (the client backs off).
- Single process: in-memory LangGraph checkpoints (paused plans are lost on restart), no authentication or
  multi-user isolation, progress is in-process. "Cross-run memory" is keyword matching over past reports; it is
  shown to the human and does not influence generation.
- Prompt-injection defences are structural mitigations with offline tests; live-model robustness is only checked
  by the manual script `tests/manual/redteam_live.py`.

## Repository layout

```text
src/research_analyst/
  agents/  pipeline/  llm/  retrieval/  grounding/  safety/  observability/  memory/  models/  evaluation/  ui/
app/main.py     Streamlit entry point (thin shim)
scripts/        run_demo.py, generate_gallery.py
tests/          unit/  integration/  ui/  (offline)  and  manual/  (live)
docs/           ARCHITECTURE.md, STORY.md
```

## Run it

Requirements: Python 3.12, [uv](https://github.com/astral-sh/uv), a Gemini API key.
Redis is optional.

```bash
uv sync
cp .env.example .env            # fill in GEMINI_API_KEY (and optionally LANGFUSE_*)

uv run streamlit run app/main.py            # the app
uv run python scripts/run_demo.py           # one run on the command line
uv run python scripts/generate_gallery.py   # regenerate the sample reports (LIVE: ~21k tokens each)
```

Quality gates (all offline — no keys, no network, no Redis): `uv run pytest` (≥ 80% coverage enforced),
`uv run ruff check .`, `uv run mypy`. CI runs these plus a Docker build.

```bash
docker compose up --build       # app on 127.0.0.1:8501 + Redis; the UI has no authentication
```

**Streamlit Community Cloud:** main file `app/main.py`, Python 3.12. Add secrets (top-level or under a section):

```toml
GEMINI_API_KEY = "..."
LANGFUSE_PUBLIC_KEY = "..."                              # optional tracing
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com"      # your Langfuse region
TOKEN_BUDGET_PER_RUN = "60000"
```

There is no Redis on Cloud (memory features do nothing) and the filesystem is ephemeral (use Langfuse for
history). `requirements.txt` is generated from `uv.lock` (`uv export --no-dev --no-hashes --no-emit-project`).

Useful settings (see `src/research_analyst/config.py` and `.env.example`): `TOKEN_BUDGET_PER_RUN`,
`LLM_PRICE_PER_1M_TOKENS`, `GEMINI_FAST_MODEL`, `GEMINI_STRONG_MODEL`, `GEMINI_STRONG_AGENTS`, `GEMINI_RATE_LIMITS`,
`GEMINI_THINKING_LEVEL`, `JUDGE_MODEL`, `QUERY_REWRITE`, `ARXIV_ENABLED`, `ARXIV_FULLTEXT`, `FETCH_PAGES`,
`READER_BATCH_SIZE`, `READER_TARGET_FACTS`, `DEBATE_MAX_ROUNDS`, `RUN_TIMEOUT_SECONDS`, `LOG_FORMAT`.
