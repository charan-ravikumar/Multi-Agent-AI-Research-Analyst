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
2. Which paid provider/model will you use next week? (Affects: native tool-calling for ReAct, judge model choice, price table `LLM_PRICE_PER_1M_TOKENS`.)
3. Should the eval set grow beyond the 5 golden queries (3 of which have never been run)?
4. Do you want the work committed on `professionalize-agent` (I haven't committed anything)?

---
*Maintenance rule: after every meaningful change or debugging session, append a dated entry under Timeline (what, why, alternatives, challenges, lessons, new interview questions). Never rewrite history; add correction notes instead.*
