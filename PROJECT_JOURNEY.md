# PROJECT JOURNEY — AI Research Analyst

Interview-prep log for this project. Read this one file before an interview.

**Conventions**
- Newest entries go at the **bottom** of the Timeline (chronological). Nothing is deleted; corrections are added as new notes.
- **[DECIDED]** = you stated it explicitly in conversation. **[INFERRED]** = I reconstructed it from code, commits or docs: confirm or correct it.
- **[MEASURED]** = a number from a real run in this repo. **[UNMEASURED]** = built but not yet evaluated. Say which one it is in an interview.
- Evidence = file paths and commit hashes, so you can open the code while you revise.

---

## 1. Project in one paragraph (the 60-second pitch)

A LangGraph pipeline that turns a question into a **cited research report**: a Planner splits it into 4–5 sub-questions (a human can
edit them), each sub-question is researched **in parallel** (search → read → synthesise), a Writer assembles the report with
**citations numbered by code, not by the model**, and an **LLM judge** checks each cited claim against the passages it cites.
Built to run on free-tier LLM APIs (Groq primary, second Groq model, Gemini fallback); Streamlit UI; Langfuse tracing; Redis memory.
It is a *staged pipeline of LLM calls*, deliberately **not** an autonomous agent swarm (the README says so).

Stack: Python 3.12, LangGraph, Streamlit, Groq (`openai/gpt-oss-20b` / `-120b`, `qwen/qwen3.8-27b` as judge), Gemini 2.5 Flash fallback,
DuckDuckGo (`ddgs`), arXiv API + PDFs (`pypdf`), `trafilatura`, BM25 (own implementation), Pydantic, Redis, Langfuse, pytest, ruff, uv, Docker.

## 2. Architecture cheat-sheet

```
START → planner → plan_approval (human interrupt) → fan_out (one Send per sub-question, parallel)
  process_sub_question [failure-isolated]: Searcher → Reader → Synthesizer
  → supervisor (fails only if EVERY branch failed) → debate (opt-in Critic⇄Reviser) → writer
  → grounding_judge (LLM, failure-isolated)  → memory_store → END
```

| Pattern | Where | Status |
|---|---|---|
| Plan → Execute | Planner → parallel branches (`pipeline/graph.py`, `nodes/branch.py`) | in use |
| Self-critique / reflection | Critic ⇄ Reviser (`agents/critic.py`, `nodes/debate.py`) | opt-in; measured not worth it by default |
| ReAct | `agents/react_search.py`, `SEARCH_MODE=react` | built 2026-11-24, **[UNMEASURED]** |
| LLM-as-judge | `grounding/judge.py`, `agents/grounding_judge.py` | built 2026-11-24, **[UNMEASURED]** vs human labels |
| Gemini-only + native tools + structured outputs | `llm/`, `agents/react_search.py`, `llm/schemas.py` | merged to `main` 2026-11-24 on your confirmation; Writer on 3.8 Flash not verified live |
| Monitor tab + streaming + redesign | `observability/runs.py`, `ui/monitor.py`, `ui/theme.py` | merged to `main` 2026-11-24 on your confirmation; streaming not yet tested live |
| Prompt versioning | `observability/prompt_store.py` (Langfuse) | merged to `main` 2026-11-24 on your confirmation; **not yet run against real Langfuse** |

## 3. Timeline

### Phase 0 — before this log (reconstructed) **[INFERRED from `git log`, README, docs/STORY.md]**
Please confirm/correct anything here that you remember differently.

- Initial multi-agent pipeline (Planner/Searcher/Reader/Synthesizer/Critic/Writer) on LangGraph; Redis memory; Streamlit UI.
- Merged branch work, added dev container, Streamlit Cloud secrets handling (`788f958`, `c9bf4ca`).
- Progress bar, curated gaps vs follow-up questions, About tab, arXiv re-ranking (`e3829e7`).
- Restructured into a `src/research_analyst/` package, LF line endings, split big modules, one shared structured-output retry helper (`7f17d67`, `3c0f6b3`, `af0ae8f`).
- arXiv full text, click-to-verify panel, Stop button, preflight model check, early-stop reading, scores, sample gallery (`cf72192`).
- Tests for those features, five real sample reports, README rewrite with **measured** numbers, architecture + failure-mode docs, engineering write-up (`5ea5608`, `bc53d9c`, `6ac3a17`).

**Problems solved in this phase** (full detail with numbers in `docs/STORY.md`; learn these — they are your best interview stories):

| Problem | Root cause | Fix |
|---|---|---|
| Runs died on a retired model, silently | 404 treated like any provider error → quiet fallback | Preflight lists real models, shows a message, scripts refuse to start |
| Eval numbers wrong at first | Gemini "thinking" tokens eat `max_output_tokens` → truncated judge JSON scored as 0 | Thinking off; truncated judge = error, not score; `finish_reason` logged |
| gpt-oss answers expensive | Reasoning tokens count | `reasoning_effort="low"` (124 → 45 completion tokens, same answer) |
| Rate-limit storms | Per-snippet Reader calls (~25/run) + 8k TPM free tier | Batched Reader (≤3 sources/call) → ~5 calls; per-model TPM limiter; jittered backoff honouring `Retry-After`; circuit breaker; provider fallback chain |
| Review loop looked useful but wasn't | Added citation markers, not faithfulness; +75% tokens; lower completeness (n=3) | Made it **opt-in**, documented the negative result |
| Report lost all citations | Model emitted `【1】` full-width brackets | `normalize_citation_brackets`; references built by code |
| Writer padded text with unsourced sentences | Prompt said "expand… exhaustive" | Prompt: use only the draft's facts. Cited-claim coverage **66% → 88%** **[MEASURED, uncontrolled before/after]** |

### 2026-11-24 — Session: add ReAct search + replace lexical grounding with an LLM judge

**Context [DECIDED]:** You only have free-tier LLM keys now and plan to get a paid API next week. You wanted every feature built *now*
so that, once the paid key arrives, only evaluation/tuning remains. Your two requests: **(1) implement ReAct, (2) replace the lexical grounding check with an LLM judge.**

**Where we started [from conversation]:** I audited the repo against your resume question. Finding: the project had Plan→Execute and
(opt-in) self-critique, but **no ReAct** — the Searcher ran a fixed recipe. I told you not to claim ReAct until it exists, and that the
cheapest credibility win was a real A/B. Hence this work.

#### A. ReAct search loop

**What was built**
- `agents/react_search.py` — `ReactSearch`: Thought → Action → Observation loop over a **closed tool set**: `search_web(query)`, `search_arxiv(query)` (only if arXiv enabled), `read_page(url)`, `finish()`. Max `REACT_MAX_STEPS` (default 4).
- `agents/searcher.py` — `SEARCH_MODE` setting (`fixed` default | `react`). Refactored the old recipe into `_gather_fixed()`; both modes feed the **same** RRF fusion → source-quality prior → per-domain cap → page/arXiv-PDF enrichment.
- `models/research_state.py` — new `react_traces` reducer field; `pipeline/nodes/branch.py` passes traces through; `ui/components.py::render_react_traces` shows "Search agent reasoning" on the results page.
- `evaluation/harness.py` — new eval config `react` (identical to `no_debate` except search mode) so the A/B is one command: `run_eval --configs no_debate,react`. `run_eval.py` patches the ReAct module onto the same record/replay fixtures.
- Tests: `tests/unit/test_react_search.py` (action parsing, step cap, duplicate query, `read_page` URL allow-list, malformed-reply handling, LLM error, injection, fallback, fixed mode unchanged).

**Technical decisions** (the "why" — interview gold). All are **[INFERRED]** design choices I made; you approved the overall goal, not each detail — tell me if you disagree with any.

| Decision | Why | Alternative & trade-off |
|---|---|---|
| ReAct only *collects candidates*; ranking/enrichment shared with fixed mode | A/B isolates the search strategy; one ranking code path to test | Letting the agent also rank/read would be more "agentic" but un-attributable gains |
| Closed tool set, JSON action format | Provider-agnostic (works with JSON mode on Groq/Gemini); no function-calling API dependency | Native tool-calling is cleaner on paid APIs but ties the loop to provider features; revisit after the paid key |
| `read_page` only accepts URLs the model was **shown** | Page text can say "now fetch http://internal…" (indirect prompt injection / SSRF steering). The fetcher is already SSRF-guarded; this is a second layer | Free-form URLs = more flexible, larger attack surface |
| Hard step cap + one corrective retry on bad action, second bad reply ends loop | Bounds cost/latency; free tiers have 8k TPM | Higher cap = more recall, more tokens |
| Fall back to the fixed recipe if the loop collects nothing | One bad model reply must not lose a branch | Fail the branch (simpler, worse UX) |
| Observations sanitised (`clean_untrusted`) and wrapped in `<observation>` tags | Web text is untrusted | — |
| Opened pages get an extra RRF "vote"; prefetched pages are not re-fetched | Treat an agent's explicit choice as weak relevance signal; save a request | Could be over-trusting the agent's choice **[UNMEASURED]** |
| Default stays `fixed` | ReAct is unproven; costs ~1 small LLM call/step (≈ 4 extra calls per sub-question) | Switch default only if the A/B shows a win per token |

**Challenges / bugs hit while building**
- Hidden coupling: tests and the eval harness monkeypatch `agents.searcher.search_web/fetch_many/search_arxiv`; the new module imports its own copies, so the eval harness had to patch both modules (done in `run_eval.py`) or ReAct would silently hit the live web — and break replay determinism.
- Lint (ruff `B023`) caught a closure-in-loop bug pattern in the judge agent; fixed with `functools.partial`.

#### B. LLM-judge grounding (replaces lexical label)

**Why [DECIDED goal / INFERRED reasoning]:** the old "strong/partial/weak" label was word overlap — it cannot tell "reduces risk by 40%" from "does not reduce risk by 40%", and the README admitted it. An LLM reading claim + passage can.

**What was built**
- `grounding/judge.py` — pure functions: batch prompt builder, **strict** verdict parser, summariser. Verdicts: `supported | partial | unsupported`, plus `supporting_refs` and a one-line `reason`.
- `agents/grounding_judge.py` — `GroundingJudgeAgent`: batches 6 claims/call (`GROUNDING_JUDGE_BATCH_SIZE`), up to 24 claims (`GROUNDING_JUDGE_MAX_CLAIMS`), uses `JUDGE_MODEL` (a *different model family* than the generator, same principle as the eval harness), JSON mode, corrective retry via the shared `generate_parsed`.
- `pipeline/nodes/grounding.py` + graph edge `writer → grounding_judge → memory_store`. Failure-isolated; refreshes `usage_summary` so the UI cost includes the judge's tokens.
- `grounding/verify.py` — extracted `best_support()` (BM25 picks *which passage to show/judge*); `verify_payload` now carries `verdicts`. Lexical score is retained **only** as a labelled fallback.
- UI: verdict + reason + "this source backs the claim" per source, summary line; honest wording ("a triage signal, not proof"). Gallery entries store verdicts (existing 5 sample reports predate the judge and show the fallback, labelled).
- Eval configs set `grounding_judge: False` so earlier results stay comparable and the eval's own judge isn't duplicated.
- Tests: `tests/unit/test_grounding_judge.py` (prompt neutralises forged tags, parser drops invalid/duplicate/foreign-id entries, batching, corrective retry, provider failure never raises, claim cap, node no-op/crash handling, full graph).

**Technical decisions** (all **[INFERRED]**)

| Decision | Why | Alternative & trade-off |
|---|---|---|
| Per-**claim** verdict + `supporting_refs`, not per (claim, source) pair | Half the output tokens; still tells the user which source backs it | Per-pair is more precise, ~2× cost |
| Judge sees only the gathered passages, "judge ONLY from the passages shown" | Measures *faithfulness to evidence* (RAG-style groundedness); stops the judge "correcting" with its own knowledge | Giving the judge web access measures truth but is costlier and non-reproducible |
| Different model family for the judge | Reduces shared blind spots / self-preference | One model for everything is cheaper |
| Batch 6 claims per call | Fits 8k TPM free tier; fewer calls than per-claim | Larger batches degrade judge accuracy (position effects) **[UNMEASURED]** |
| Never fail the run if the judge fails; fall back to lexical, disclosed | The report is already done; the judge is an add-on | Hard-fail = worse UX; silent fallback = dishonest (the same lesson as the retired-model story) |
| BM25 still selects the passage | Retrieval ≠ judgement; cheap, deterministic | Embedding retrieval = extra dependency (README: no vector store, by design) |

**Honest limits (say these in the interview):** the judge is itself an LLM; it has **not been validated against human labels**; it sees extracted passages, not the full page; ReAct's effect on quality/cost is **unmeasured**. Existing numbers (88% cited-claim coverage etc.) predate both features.

**Verification:** ruff clean; full suite **313 passed** offline (no live LLM calls — free-tier quota was not spent). Nothing was run against a real provider this session.

#### C. Housekeeping
- `README.md`, `docs/ARCHITECTURE.md`, `.env.example`, About tab updated to describe both features and their unmeasured status.
- Nothing committed yet (working tree only).

### 2026-11-24 — Feature branch `feature/langfuse-prompt-management`: Langfuse Prompt Management **(merged to main on your confirmation)**

**Context [DECIDED]:** you asked where prompt versioning could live; I recommended in-repo versioning first, you chose **Langfuse Prompt Management**. You also set the workflow: new features go on a branch and merge to `main` only after you confirm.

**What was built**
- `observability/prompt_store.py` — `register(name, text, formatted=)` declares each SYSTEM prompt in the repo (the fallback); `resolve(name)` returns the Langfuse text for `LANGFUSE_PROMPT_LABEL` (default `production`) with its version, or the repo text. 9 prompts managed: planner, reader, synthesizer (+ revision), critic, writer, query-rewrite, react-search, grounding-judge. USER templates stay in code.
- `agents/base.py::llm_generate` — looks up which managed prompt a system message came from and passes the Langfuse prompt object to the generation, so each trace is **linked to the exact prompt version**; metadata records name/source/version. The old SHA-1 `version` tag is kept.
- `observability/prompt_sync.py` + `scripts/sync_prompts.py` — pushes repo prompts as new Langfuse versions (`--label`, `--dry-run`, idempotent: identical text at that label is skipped).
- `evaluation/run_eval.py` — result JSON now records `prompt_label` and `prompts_used` (so an eval is reproducible/attributable to prompt versions).
- Settings: `LANGFUSE_PROMPTS` (kill switch), `LANGFUSE_PROMPT_LABEL`, `LANGFUSE_PROMPT_CACHE_SECONDS`. Tests: `tests/unit/test_prompt_store.py` (16 tests; suite 329 passing offline, ruff clean).

**Technical decisions** (all **[INFERRED]** unless stated)

| Decision | Why | Alternative & trade-off |
|---|---|---|
| Repo text is always the fallback; Langfuse is an *override* | App must work offline/in CI/with no keys; git stays reviewable | Langfuse as sole source = single source of truth but a hard runtime dependency |
| Reject a remote prompt whose `.format` placeholders differ from the code's | A UI edit (e.g. renaming `{min_q}`) would otherwise crash every run with `KeyError` | Trust the UI = flexibility, outage risk |
| After one Langfuse failure/miss, skip remote for 60 s | Avoid paying a 3 s timeout on every call when Langfuse is down/prompt absent | Per-call retry = fresher, slower |
| Link generations via `prompt=` + keep hash tag | Langfuse then shows per-version latency/cost/scores; hash still works without Langfuse | — |
| Only SYSTEM prompts managed | USER templates are coupled to code-supplied data; versioning them separately invites mismatches | Manage both = full UI control, more drift risk |
| Label-based selection (`production`/`staging`) | Roll forward/back and A/B in eval by changing a label, no deploy | Pin integer versions = reproducible, less convenient |

**Challenges:** circular imports (the react/judge modules import the store, so the store can't import them) → registration happens next to each prompt at import time and the sync script imports the modules; Langfuse text prompts use `{{var}}` while ours use Python `{var}` → we fetch the raw text and format ourselves, and validate placeholders.

**Honest limits:** verified with a fake Langfuse client only — **not run against real Langfuse** (needs your keys). Open items: run `scripts/sync_prompts.py --dry-run` then real, confirm traces show linked prompt versions in the UI.

**Interview questions this adds:** Why keep a local fallback? How do you stop a prompt edit in a UI from breaking production? How would you A/B two prompt versions and attribute the difference? What's the difference between tagging a trace with a prompt hash and linking it to a managed version? Where do prompts belong — code or a registry — and what are the trade-offs (review, rollback, drift, latency)?

### 2026-11-24 — Feature branch `feature/gemini-only-native-tools`: Gemini-only, native function calling, structured outputs **(merged to main on your confirmation)**

**Context [DECIDED — your instructions]:** remove Groq; use only Gemini; **no fallback**; `gemini-3.5-flash-lite` for most agents and `gemini-3.8-flash` only where needed; function calling through the API; structured outputs for all tools/functions. Your limits: 3.8 Flash = 5 RPM / 250K TPM / 20 RPD; 3.5 Flash Lite = 15 RPM / 250K TPM / 500 RPD.

**What was built**
- **LLM layer rewritten** (`llm/`): Groq client and the provider-fallback chain deleted. `client.py` maps tier→model, enforces limits (waits for RPM/TPM; **persisted per-day counter** that raises `DailyQuotaExceeded` instead of burning retries; resets midnight Pacific), retries 429/5xx with jittered backoff, and a per-model circuit breaker that now **fails fast** (no other model to try). `gemini_client.py` is the only module that talks to the API.
- **Which agents use the strong model [INFERRED]:** only the **Writer** (`GEMINI_STRONG_AGENTS=writer`, configurable). 20 RPD ⇒ ~10-20 runs/day at 1-2 Writer calls per run; adding more agents would exhaust it quickly. Everything else (planner, reader, synthesizer, critic, query rewrite, ReAct, grounding judge) runs on Flash Lite.
- **Structured outputs** (`llm/schemas.py` + `BaseAgent.response_schema`): Pydantic schemas passed as Gemini `response_schema` for planner, reader, synthesizer (+revision), critic, query rewrite, grounding judge, eval judge. The API guarantees the **shape**; the agents still validate the **meaning** (verdict ids belong to the batch, source ids valid, sub-question count) and keep the corrective retry. The Writer stays plain Markdown by design.
- **Native function calling for ReAct** (`agents/react_search.py` rewritten, `llm/tools.py`): tools declared as function declarations with JSON-schema parameters, `FunctionCallingConfig(mode=ANY)` so the model can only answer with a tool call, automatic execution disabled, every tool has a required `thought` argument (keeps it ReAct), the model's own turn is passed back untouched (Gemini 3 **thought signatures**), parallel calls in one turn are all answered (capped at 3). The earlier JSON-in-text action parser was removed; `validate_call` re-checks arguments as defence in depth.
- Preflight now checks Gemini model names (`check_models`); UI, secrets, scripts, Docker/requirements (groq removed from `pyproject.toml`/`uv.lock`), README/ARCHITECTURE/.env.example updated. Tests: 343 passing offline, ruff + mypy clean.

**Technical decisions** (INFERRED unless stated)

| Decision | Why | Alternative & trade-off |
|---|---|---|
| Client-side RPM/TPM waiting + persisted RPD counter | Free tiers fail with 429s that retries can't fix (daily cap); fail early with an actionable message | Rely on 429 handling = simpler, wastes quota and time |
| Count every retry attempt against RPD | Pessimistic but safe; Google may or may not count 5xx | Count only successes = risk of exceeding the real limit |
| Strong model for the Writer only | Final report quality matters most; fewest calls | Planner/Synthesizer on strong = better reasoning, quota gone fast |
| `thinking_level` setting, default unset | Probe showed `thinking_budget=0` is **rejected (400)** by Gemini 3.x; level is accepted on Flash Lite | Always set a level = fewer thinking tokens, but unverified on 3.8 |
| Judge defaults to the fast model | 3.8 quota is too scarce; but the judge is now **the same model family as the generator** → possible self-preference bias | Use 3.8 as judge = more independent only in size, not family |
| Circuit breaker fails fast with no fallback | Avoid every parallel branch paying full retries during an outage | Keep retrying = slower failures |
| Writer output not schema-constrained | A single long Markdown string gains nothing from JSON | Wrap in `{report: "..."}` = escaping overhead |

**Live verification (3 probes + smoke test on your real key) [MEASURED]:** structured output with a Pydantic schema works on Flash Lite; native function calling in mode ANY works and returns thought signatures; the full Planner agent returned 4 sub-questions via structured output; a real ReAct run executed 3 tool-call turns and collected 28 candidates. **`gemini-3.8-flash` returned 503/504 on every attempt I made** (about 9 requests, ~3 of your 20 daily requests used by the smoke test's retries plus earlier probes) — so the **Writer on 3.8 is NOT verified live**. This looked like provider overload, not a code bug, but treat it as an open risk: with no fallback, a persistent 3.8 outage fails the run. Mitigation: `GEMINI_STRONG_AGENTS=` (empty) runs everything on Flash Lite.

**Challenges / lessons**
- `thinking_budget=0` (used for Gemini 2.5) 400s on Gemini 3 models → switched to an optional `thinking_level`; found by a live probe, not docs.
- Function-calling history can't be rebuilt from text: with Gemini 3 the model's turn (thought signatures) must be returned unchanged, so the conversation history is provider-native and opaque (`llm/tools.py`).
- Evaluation harness: ReAct model turns are **not LLM-cached** (provider objects), only searches/pages are replayed → ReAct A/B is repeatable in evidence, not in model decisions.
- Lost provider independence: the earlier eval used a different-family judge (qwen). Now generator and judge are the same family — a limitation to state in interviews.

**Honest limits:** README measurements (88% cited claims etc.) are from the Groq era and are **not re-measured** on Gemini; quality of Flash Lite for synthesis is unproven; the RPD counter is local (not shared between machines).

**Interview questions this adds:** Native tool calling vs prompting for JSON: what does the API guarantee and what must you still validate? What is `mode=ANY` and why use it? What are thought signatures and what breaks if you drop them? How do you design for hard free-tier limits (RPM/TPM/RPD) — wait, fail fast, or degrade? Why no fallback provider, and what are the risks? What do you lose when generator and judge share a model family? Structured output (constrained decoding) vs parse-and-retry: trade-offs?

### 2026-11-24 — Feature branch `feature/monitoring-streaming-ui`: Monitor tab, streaming answer, redesign **(merged to main on your confirmation)**

**Context [DECIDED — your requests]:** (1) a small monitoring page in the app; (2) stream the answer on screen ("does it stream? if not add it" — it did **not** stream before: the Writer returned the whole text at once); (3) make the page look distinctive, not like a generic AI-generated page. Before this, you asked what a production-grade project would track; I listed the metric families (traceability, reliability, latency, cost, quality, agent behaviour, safety, eval drift) and you chose "a small monitoring page in the app" (no Prometheus/Grafana yet).

**What was built**
- **Per-run record** (`observability/runs.py`): one JSON line per run (status complete/partial/failed/cancelled, error kind, elapsed, tokens/calls by agent and model, model time per agent, cited ratio, judge verdict counts, branch failures, coverage warnings, ReAct step stats, config snapshot incl. models and prompt versions). Written from `ui/runner.py` in a `finally`, so failures and Stop clicks are recorded too. No report text is stored.
- **Process counters** (`observability/metrics.py`) incremented in `llm/client.py` (successes, retries, errors by HTTP status, seconds waiting on rate limits, breaker opens) and in `generate_parsed` (structured-output corrective re-asks per agent). `llm.client.quota_status()` exposes today's per-model request use vs limits. `UsageTracker` now also sums latency per agent/model.
- **Monitor tab** (`ui/monitor.py`): headline ledger (runs, completed %, median time + p95, median tokens, claims supported %, claims cited %), free-tier quota bars, reliability (status breakdown, failure causes, counters), run-time chart, tokens and model-time by agent, verdict share per run, recent-runs table, JSONL download.
- **Streaming**: `gemini_client._stream` uses `generate_content_stream` and calls `on_text` with the cumulative text (a retry restarts from an empty string; a broken callback never fails generation). Plumbed through `llm.generate(on_text=)` → `BaseAgent.llm_generate` → Writer → `progress.stream()`. The researching page is now an `st.fragment(run_every=0.7s)` that re-renders itself (no full-page rerun, no flicker) and shows a "live draft" panel.
- **Redesign** (`ui/theme.py`, `.streamlit/config.toml`): "field notebook" — warm paper, ink, one vermilion accent; Fraunces (display), Newsreader (reading), Hanken Grotesk (controls), JetBrains Mono (labels/numbers); newspaper masthead with double rule, mono tab labels, square ink-bordered buttons with an offset-shadow hover, ledger-style metrics, the report in an editorial card. Emoji chrome removed (status colours use `:green[●]` etc.). Checked visually with headless Chrome screenshots against a seeded run history.
- Tests: 374 passing offline (`tests/unit/test_runs_and_metrics.py` adds 31: record shape, classification, error buckets, storage, aggregation, recording from the thread for complete/failed/cancelled runs, counters, quota status, streaming, Monitor page empty/with data); ruff and mypy clean.

**Technical decisions** (INFERRED unless stated)

| Decision | Why | Alternative & trade-off |
|---|---|---|
| JSONL file for run history | zero infrastructure, human-readable, append-only; fine for a demo-scale app | SQLite = queryable and safe under concurrency; Postgres/Prometheus = real production monitoring |
| Counters in memory, not persisted | they describe the shared client's behaviour, not one run; resetting on restart is stated on the page | Persisting/pushing to Prometheus gives history across restarts |
| Stream only the Writer | it is the one long, user-visible generation; other agents return JSON that is useless half-finished | Streaming every section = more calls held open, little value |
| Stream raw text, replace with the verified report at the end | citation numbering/reference list are computed in code after generation; streaming them would require a streaming parser | Delay display until validated = no streaming |
| `st.fragment(run_every)` instead of `sleep + st.rerun()` | partial reruns avoid full-page flicker and re-running the whole script every 0.7 s | Full rerun polling is simpler but visibly jumpy |
| Judge/cited metrics read from the run record, not recomputed | the page never calls an LLM and works on history | — |
| Custom CSS on Streamlit's DOM | the only way to get a distinct identity inside Streamlit | Fragile to Streamlit upgrades (class/test-id names change); a React front end would be sturdier |

**Challenges / lessons**
- CSS selectors written for older Streamlit silently did nothing (tabs and buttons are `div[data-testid=stTab]` and `[data-testid^=stBaseButton-]` in 1.59). Found only by looking at screenshots and inspecting the DOM — **verify visual work by rendering it**. Also: Streamlit here has no file watcher (watchdog missing), so a running dev server does not pick up edits to imported modules; restart it.
- Test doubles that predate `on_text` broke 58 tests; fix was to pass `on_text` only when streaming (production code) and teach the fakes to accept it, instead of changing every test.
- A shared temp file made tests order-dependent (run history, quota counter, counters) — each is now reset per test in `conftest.py`.

**Honest limits:** the streaming path was tested with fakes and a scripted stream, **not against the live API**; the Monitor charts were screenshot-tested with seeded sample data, not real runs; the live-draft panel has no screenshot because it needs a real run; run history on Streamlit Cloud is ephemeral; "time per stage" is model time summed per agent, not wall-clock.

**Interview questions this adds:** What would you monitor for an LLM application, and why those metrics? How do you separate per-request telemetry from process-level counters? What are the trade-offs of storing metrics in a file vs a TSDB? How does streaming interact with output post-processing (validation, citation numbering)? Why `st.fragment` for live updates? How do you test UI code that depends on a framework's internal DOM? What does "completed" mean as a success metric when partial reports exist?

### 2026-11-25 — First real end-to-end run + fixes: branch `fix/overload-backoff-and-run-usage` **(merged to main 2026-11-25 together with the two-key pool)**

**Context [DECIDED — your request]:** "run and check if everything works fine." I ran the app locally with your real key and drove it in headless Chrome (type a question → plan → approve → watch the run → results → verify panel → Monitor), after first loading your deployed Streamlit link (loads, new design, no errors; Monitor empty as expected).

**What I observed [MEASURED]**
- **Run 1 (defaults): FAILED at the Writer.** Planning, search, reading and synthesis worked (~2.5 min, 21 calls) and then `gemini-3.8-flash` returned **503 on all retries** → whole run failed, discarding all the research. Four direct probes over several hours: 3.8 Flash 503'd every time, Flash Lite answered every time. 3 retries × 2 runs also burned ~6 of your 20 daily 3.8 requests.
- **Run 2: planner failed** — Flash Lite itself 503'd for ~10 s (a transient spike); recovered within a minute.
- **Run 3 (`GEMINI_STRONG_AGENTS=` empty, everything on Flash Lite): completed, as a *partial* report** — 2 of 4 sub-questions failed on 503 spikes and the judge stopped after one batch (12 of the claims checked: 11 supported, 1 partial). What worked: **the report streamed live** (3,894 chars seen before completion, screenshot confirmed), citations 14/15, verify panel with judge verdicts, no UI exceptions, Monitor tab populated with real data.

**Bugs found and fixed here**
1. **Run record had 0 tokens/calls** — root cause: the memory node clears the in-memory usage tracker before the UI thread records the run, so the record read an empty tracker. Fix: prefer the run's own final `usage_summary`; fall back to the tracker only for runs that died before producing one.
2. **503 backoff far too short** (1.5 s + 3.8 s ≈ 5 s total) for Google "high demand" spikes that last tens of seconds, so retries just burned requests and branches failed. Fix: for 5xx the wait is `LLM_OVERLOAD_BACKOFF_SECONDS` (8 s) × attempt with jitter, and `LLM_MAX_RETRIES` 3→4 (≈ 8+16+24 s). 429 keeps the short exponential backoff.
3. Judge line said "(default)" instead of the model name — now shows the real model.
4. My own earlier test runs had written 8 junk records into your local (gitignored) `logs/runs.jsonl` before I isolated tests; removed (kept the 2 real ones).

**Still open — a decision for you [not made]:** the Writer on `gemini-3.8-flash` fails every run while that model is overloaded, and with no fallback that fails the whole run. Options: (a) set `GEMINI_STRONG_AGENTS=` empty (everything on Flash Lite) until 3.8 stabilises; (b) keep it and accept failures; (c) allow the Writer to retry on the fast Gemini model if the strong one is down (a model fallback inside Gemini, which you said you didn't want, so I did not add it).

**Lessons:** a happy-path unit-test suite (374 tests) did not reveal that a provider outage at the last step throws away the whole run, nor the tracker/record ordering bug — both appeared only in a real end-to-end run. Retry policy must match the failure's time scale (overload spikes ≠ rate limits).

**Interview questions this adds:** How do you handle a provider that is down only for your most important call? What is the right retry/backoff for 429 vs 503? How do you test timing/ordering bugs between components that clear shared state? Why partial results over failure, and how do you tell the user?

### 2026-11-25 — Feature branch `feature/two-key-llm-pool`: two Gemini keys, split in the wrapper **(merged to main on your instruction, together with the fix branch below)**

**Context [DECIDED — your instructions]:** you added `GEMINI_API_KEY_2` from a different free-tier account; use both, plan so rate limits are not hit, 3.5 Flash Lite for most and 3.8 Flash only where a better model is needed, split calls across the keys **in the LLM wrapper only**, run once to test, then push to main. You also said: **never add Claude as a co-author** (applied from this commit on; earlier commits on main still carry the trailer).

**What was built (all in `llm/` + config; agents/tracing/UI unchanged except display)**
- `settings.gemini_keys` → `[("k1", secret), ("k2", secret)]`; `gemini_client` keeps one SDK client per key and takes `api_key`.
- `client._choose_key`: picks the key with the least wait for its per-minute budget, then the most requests left today, with a rotating tie-break so equal keys alternate; skips keys whose daily budget is spent or whose breaker is open; a **retry prefers the other key** (first retry is immediate when two keys exist, because the other account is a separate quota).
- Per-(key, model) quota counter (file format now `{"k1": {model: n}, "k2": ...}`; the old single-key file is read as k1), rate limiter, and circuit breaker (one failure counted per exhausted *call*, not per attempt).
- **Planning below the limits:** `GEMINI_LIMIT_HEADROOM=0.9` on RPM/TPM (5 → 4/min for 3.8 Flash, 15 → 13/min for Flash Lite) and `GEMINI_DAILY_RESERVE=0.05` on the daily budget (20 → 19, 500 → 475), per key.
- Usage carries `key_id`; `UsageTracker.by_key` and the run record's `by_key` show the split; Monitor shows combined limits and the per-key split. Metrics are labelled by key id.
- Tests: 392 passing (12 new for the pool: spread, prefer the key with more left, skip exhausted, retry on the other key, breakers, headroom, doubled per-minute capacity, status aggregation, old quota file, one SDK client per key, secrets never in labels).

**Technical decisions** (INFERRED unless stated)

| Decision | Why | Alternative & trade-off |
|---|---|---|
| Least-wait-then-most-remaining key choice | avoids waiting when the other key is free, and drains both accounts evenly so neither hits its daily cap first | Plain round-robin ignores that one key may be nearly spent or rate-limited |
| Retry prefers the other key, first retry immediate | 429/quota/503 on one account is often not shared by the other; saves 8–16 s | Always back off = slower; ineffective for model-wide overload |
| Headroom (0.9 / 5%) | provider-side counting windows differ from ours; stopping short avoids 429s which also burn quota | 1.0 = max throughput, occasional 429 |
| Breaker counts per exhausted call, per (key, model) | a single call's four attempts must not open the breaker on their own | Per-attempt counting opened it mid-call (found by a test) |
| Keys known only in `llm/` | you asked for this; keeps agents testable and provider-agnostic | — |
| Writer stays the only 3.8 Flash agent | scarce quota (now 2×19/day) | add Planner (+1 call/run) if 3.8 is stable |

**Live verification [MEASURED]:** Run A (defaults) failed at the Writer — `gemini-3.8-flash` returned 503 on **both** keys, 4 attempts (k2, k1, k2, k1), so it is a model-wide outage on Google's side, not key-specific; before that the pool split 17 calls 7/10 and recovered three Flash Lite 503s by switching key. Run B (Writer on Flash Lite via `GEMINI_STRONG_AGENTS=`): **complete**, 26 calls split **13/13**, 0 retries, 39 s of pipeline time (vs ~158 s on one key with waits), 41/41 report sentences cited, grounding judge 24 claims (22 supported, 2 partial), live draft streamed (8.6k chars), no UI exceptions.

**Open item:** with the default `GEMINI_STRONG_AGENTS=writer`, runs fail at the Writer while 3.8 Flash is down (it has failed every call for ~a day). Setting `GEMINI_STRONG_AGENTS=` (empty) in `.env` / Streamlit secrets runs everything on Flash Lite. I did not change the default; your call. A cheaper alternative that keeps "no fallback" in spirit is also possible: persisting the finished research so a Writer failure can be retried without redoing 25 calls.

**Interview questions this adds:** How do you spread load across multiple API keys/accounts safely (and is that within provider terms)? How do you schedule between resources with different remaining budgets? Why count breaker failures per call? How do you keep the secret out of logs/metrics? What is the difference between a per-key and a model-wide failure, and how does your retry policy tell them apart?

### 2026-11-25 — Feature branch `feature/redis-durable-runs`: durable runs on Redis (P0) **(to be merged to main on your instruction)**

**Context [DECIDED — your instructions]:** after a Redis integration analysis you asked to build P0 (queue, workers, durable checkpoints): branch, build, test, commit, merge, push. You added Redis Cloud keys to `.env`.

**Finding:** `REDIS_USER_API_KEY` / `REDIS_ACCOUNT_KEY` are Redis Cloud *management API* keys, not a database connection. A read-only API check showed the account has **no database**. The app needs `REDIS_URL`. Nothing was created in the cloud; everything was verified on a local Redis 8.6.

**What was built** (`jobs/`, `ui/queued.py`; `EXECUTION_MODE=inline` stays the default) [INFERRED — design is mine]
- Redis Streams job queue with a consumer group; run records as hashes; every state+stream change is one Lua script.
- Own LangGraph checkpointer on core Redis commands (msgpack with a model allow-list, TTLs, own thread pool).
- Worker: heartbeat **thread**, lease with random token, `XAUTOCLAIM` takeover, resume from checkpoint, cancel flag, graceful hand-back on SIGTERM, dead-letter after 3 deliveries.
- UI submits/polls/approves/cancels; run id in the URL so a reload reattaches.
- Idempotency: report record id derived from the session id; double submit/approve ignored. Token usage is persisted so budgets and totals survive a resume.

**Decisions**

| Decision | Why | Alternative & trade-off |
|---|---|---|
| Own checkpointer instead of `langgraph-checkpoint-redis` (changes my earlier recommendation) | the package needs RedisJSON + query engine; Homebrew Redis and `redis:7` lack them; core commands also run on fakeredis | ~270 lines I must maintain; mitigated by LangGraph's conformance suite (all base capabilities pass; no copy/prune extensions) |
| Streams, hand-written | long jobs need heartbeats and takeover; `arq` is maintenance-only | Celery/Taskiq: less code, less control with 15-minute async jobs |
| Heartbeat in a thread | a full default executor (LLM calls sleeping on rate limits) or busy loop must not expire the lease | a coroutine is simpler but starvable |
| Application errors are not retried; only worker death / Redis errors redeliver | LLM layer already retries; re-running a failing run burns quota | automatic job retry |
| Resume decided from the checkpoint, not the job | one code path for first attempt and redelivery | storing a cursor in the job duplicates truth |

**Verification [MEASURED]**
- 438 offline tests, 92% coverage; job and checkpointer tests run on fakeredis **and** a real server; a test `kill -9`s a real worker process.
- Live run (real Gemini, web search): plan 4.6 s; killed worker at 2/4 branches (11 calls, 7.5k tokens spent); second worker took over after ~11 s (10 s lease); the 2 finished branches ran once, the 2 unfinished ran again; complete report, 24 calls, 27.4k tokens, 67 s, judge 19 supported / 5 partial / 0 unsupported of 24. About 340 KB Redis memory for the run. n=1.

**Bugs found**
1. Live run: `Cannot send a request, as the client has been closed` — two parallel first calls each built a Gemini client; the discarded one was closed while in use. Pre-existing. Fixed with a lock (test failed before the fix).
2. Live run: the UI showed only the last attempt's elapsed time after a resume. Fixed (wall clock from first research start).
3. CI's mypy step was already failing on `main` (`react_traces` missing in two scripts). Fixed by a shared `new_research_state`.

**Still open [not done]**
- Rate limiter is per process (P1): run ONE worker. Seen live: the replacement worker got two 429s because it did not know what the dead one had just spent.
- Stop / takeover cannot interrupt LLM calls already queued in threads.
- At-least-once: an interrupted branch is paid for twice; checkpoint writes are not fenced by the lease.
- Redis memory: ~80 runs fill 30 MB within the 7-day TTL.
- Docker/Compose changes not run locally (no Docker); not tested against a hosted Redis.

**Interview questions this adds:** Why Streams over Pub/Sub or lists? What delivery guarantee do you have and where can a duplicate happen? How does a worker know it is a zombie, and what can it still write? Why is resume cheap in LangGraph (pending writes)? Why did you write a checkpointer, and how do you know it is correct? How do you test crash recovery?

### NEXT — when the paid key arrives (plan, not yet done)
1. Smoke-test both features live on one query (`SEARCH_MODE=react`), read the Langfuse trace of a ReAct run.
2. **A/B:** `run_eval --queries 5 --configs no_debate,react` (≥3 runs each for variance); report faithfulness, completeness, tokens, calls, time.
3. **Validate the judge:** hand-label ~40 claims (supported/partial/unsupported), report agreement (accuracy / Cohen's κ), and judge-vs-lexical agreement. This turns "I used an LLM judge" into "my judge agrees with humans X%".
4. Consider native tool-calling for ReAct; tune `REACT_MAX_STEPS`; decide the default `SEARCH_MODE` from the data.
5. Regenerate the sample gallery so reports carry judge verdicts.

## 4. Concepts to be able to explain

- **ReAct** (Yao et al. 2022): interleave reasoning and tool use; observation feeds the next thought. vs **Plan-and-Execute** (plan up front, fixed steps) — what each is good at; why this project uses both at different levels (plan across sub-questions, ReAct within one).
- **Self-critique / reflection** and why a loop can raise surface metrics (citation markers) without raising faithfulness.
- **LLM-as-judge**: biases (self-preference, position, verbosity), why a different model family, why you must validate against humans, why a truncated judge must be an error not a score.
- **Groundedness vs correctness**: supported-by-evidence is not the same as true.
- **Reciprocal Rank Fusion**, BM25, source-quality priors, per-domain diversity caps.
- **Indirect prompt injection** and structural mitigations (delimiter stripping, forged-marker removal, closed tool set, URL allow-list, SSRF guard).
- **Reliability engineering for LLM apps**: retry with jitter + `Retry-After`, circuit breaker, TPM limiter, provider fallback, token budget, failure isolation per branch, honest degradation.
- **LangGraph**: `Send` fan-out, reducers (`operator.add`), checkpointer + `interrupt()` for human-in-the-loop, why one composite branch node (Send arg only reaches the first node).
- **Evaluation design**: record/replay fixtures for identical evidence, cached LLM calls, paired comparison, why n=3 is not significant.

## 5. Likely interview questions (with where the answer lives)

1. Is this an agent? Where is the agency? *(pipeline vs ReAct mode; be precise — fixed pipeline by default.)*
2. Walk me through ReAct in your code. What stops it looping forever or being steered by a web page? *(step cap, closed tools, URL allow-list, sanitised observations)*
3. Why did you make ReAct opt-in? How would you decide to make it default? *(A/B on identical evidence; per-token gain)*
4. How do you know your LLM judge is right? *(honest answer: not yet validated; plan = human-labelled set + agreement/κ)*
5. Why replace the lexical check? Give a failure case of word overlap. *(negation, numbers)*
6. Why a different model for the judge? What biases remain?
7. How do you stop the model inventing citations? *(numbers assigned by code; invented markers removed; references built from cited sources)*
8. Your review loop didn't help. How did you find out and what did you do? *(eval, n=3 caveat, made opt-in)*
9. How did you handle free-tier rate limits? *(batched Reader, TPM limiter, backoff, breaker, fallback chain)*
10. What happens when one branch fails? When the judge fails? When the provider returns truncated JSON?
11. How would you scale this to many users? *(durable checkpointer, job queue, auth, per-user budgets — README lists these as gaps)*
12. What would you measure first with a bigger budget, and what result would change your design?
13. Why no vector store? *(small evidence set per sub-question; BM25 cheap/deterministic/testable)*
14. How did you test LLM code offline? *(scripted `FakeLLM` dispatching on system prompt; record/replay fixtures)*
15. What are the cost drivers per run? *(~22 calls, ~21.5k tokens median [MEASURED]; ReAct adds ~4 calls/sub-question; judge adds ~4 calls)*

## 6. Open questions for you (I won't assume — please answer when convenient)

1. Phase 0 above is reconstructed from git history. Which parts did **you** design vs. extend? (Needed so the "decided vs inferred" labels are right.)
2. ~~Which paid provider/model?~~ Superseded 2026-11-24: you chose Gemini-only (Flash Lite + 3.8 Flash). Still open: will the paid plan lift the RPD limits, and should `GEMINI_RATE_LIMITS` be updated?
3. Should the eval set grow beyond the 5 golden queries (3 of which have never been run)?
4. Do you want the work committed on `professionalize-agent` (I haven't committed anything)?

---
*Maintenance rule: after every meaningful change or debugging session, append a dated entry under Timeline (what, why, alternatives, challenges, lessons, new interview questions). Never rewrite history; add correction notes instead.*
