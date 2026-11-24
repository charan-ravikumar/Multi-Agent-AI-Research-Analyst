# What broke when I ran a research agent on free-tier APIs

*Build notes for anyone reading the repo. Everything here was measured on this project; the numbers are
small-sample and I say so where it matters.*

The project is a LangGraph pipeline: plan → search → read → write a cited report. It worked in a demo. Then I
ran it for real, on free API keys, and most of what I learned was about the edges between my code and the
services it depends on. Eight things, roughly in the order they bit me.

## 1. The model I built on was retired, and nothing told me

Groq retired the Llama models the project was written against. Every call returned HTTP 404 "model does not
exist". My client treated a 404 like any provider failure and quietly routed everything to the fallback —
Gemini. A run would start, burn through the fallback's allowance, and die somewhere unrelated.

**Fix:** a preflight check that asks the provider which models exist (cached for 10 minutes) and turns the
problem into a message on the start page: *"openai/gpt-oss-120b is not available for this key; available chat
models are …"*. Scripts refuse to start rather than spend quota. The lesson: a fallback that works *silently*
hides the failure it exists to survive.

## 2. Hidden reasoning tokens eat your output budget

Two different models, same trap:

- **Gemini 2.5 Flash** is a "thinking" model. Its hidden reasoning counts against `max_output_tokens`. My judge
  call asked for a small JSON object with a 1,500-token cap; it returned `finish_reason=length` after 3,897
  tokens and a truncated, unparseable object. Because my harness scored unparseable output as zero, the first
  evaluation numbers were wrong. Thinking is now off by default, and a truncated judge is an error, not a score.
- **gpt-oss** models count reasoning tokens too: the same JSON answer cost 124 completion tokens at default
  effort and 45 at `reasoning_effort="low"`.

## 3. Free tiers have hard, undocumented edges

Mid-evaluation, Gemini returned `429 … generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash`.
**Twenty requests per day.** That broke two assumptions at once: Gemini is not a usable fallback for a
multi-call pipeline, and my "independent judge" (Gemini scoring Groq's output) had silently fallen back to the
generator's own model family for one query. I now record the judge's model *family*, exclude any run where it
matches the generator from the comparison, and put a second Groq model (different family, separate quota) in
the middle of the fallback chain.

## 4. My own rate limiter was the bottleneck

To stay under Groq's 8,000 tokens/minute I added a sliding-window limiter that reserved `max_tokens` for every
call before it ran. Real completions were a few hundred tokens, so each call waited ~55 seconds for budget it
never used. A run took 313 s. Reserving a realistic estimate and settling to actual usage afterwards brought the
same run to 84 s. Measure your safety mechanisms; they are code too.

## 5. The review loop didn't earn its cost

The headline "agentic" feature was a Critic ⇄ Reviser loop. I evaluated it properly: 3 questions, recorded
search results so both configurations saw identical evidence, a judge from a different model family.

| 3 queries, 1 run each | no loop | with loop |
|---|---|---|
| claim sentences that carry a citation | 79% | 91% |
| judge faithfulness | 0.75 | 0.78 |
| judge completeness (1–5) | 4.3 | 3.7 |
| tokens per run | 15.4k | 26.9k |

It mostly *adds citation markers*; faithfulness is flat, completeness drops (the reviser prunes), it costs ~75%
more, and it hit its round cap on all three. It is now opt-in. n=3 proves nothing statistically — but it is
enough to stop claiming a benefit I haven't shown. Earlier, the unfiltered loop *never* converged because the
Critic always returned its maximum of 10 objections; making objections carry a severity and letting only
"high" ones block acceptance fixed that part.

## 6. "Unresolved gaps" was mostly noise

Each sub-question asked the model to list what the evidence did not cover. A real run produced **20 items for
5 sub-questions, none of them detected by the system** — generic, unverified, repetitive ("user acceptance and
training requirements for clinicians"). I split it in two: *coverage warnings* the system genuinely detects (a
search found nothing, a branch failed, no usable evidence) and a short deduplicated list of *open questions*
that become one-click follow-up research. In the same run: 9 raw items → 1 warning + 6 follow-ups.

## 7. The Writer ate the citations — twice

Two separate live failures, neither visible to any unit test:

- **`【1】` instead of `[1]`.** The gpt-oss Writer sometimes emits citations in full-width brackets. My pipeline only
  recognised ASCII `[1]`, so every citation in one report vanished, the code-built reference list came out empty
  ("no claim carries a citation") — and the grounding stats still said 19 of 21 sentences were cited, because they
  were measured on the *draft*. Fixes: normalise bracket styles (`【1】`, `[1†source]`, `[1, 2]`, `[1-3]`) at every
  LLM boundary; make the Writer retry once with an explicit message when it drops all markers; measure citation
  coverage on the *final* text; and add a CI quality gate that fails if any bundled sample report has no resolvable
  citations.
- **Padding.** The Writer prompt said "expand on each point … exhaustive, elaborate". Measured on the final text,
  only 66% of claim sentences were cited versus ~81% in the draft: the Writer was adding unsourced elaboration.
  Telling it to use only the draft's facts and keep the markers brought the same five questions to 88%.
  (Before/after on different search results, so treat it as a direction.) A prompt that asks for "more" is a
  grounding bug.

Related: Groq's JSON mode can reject a model's answer server-side (HTTP 400 `json_validate_failed`) but hands the
rejected text back in the error. My client treated the 400 as "this model failed" and paid for a second
generation on the fallback model; it now passes the returned text to the tolerant parser first.

## 8. The boring integration failures

- Langfuse returned 401 for correct keys — they were for the US region; the SDK reads `LANGFUSE_BASE_URL`,
  my config only read `LANGFUSE_HOST`.
- Streamlit Cloud exports only top-level secrets as environment variables; keys pasted under a `[section]`
  header were invisible. The app now flattens `st.secrets` itself and shows a clear error if no key is set.
- arXiv rate-limits by IP (HTTP 429). The client spaces requests, backs off for two minutes, and caches.

## What is still unproven

- The evaluation predates query rewriting, source ranking, arXiv PDF text, early-stop reading and the Writer prompt
  change; their end-to-end effect on answer quality is unmeasured (the 66% → 88% citation ratio is uncontrolled).
- Citation checks are lexical (shared key terms), not semantic. "Strong support" means the source uses the
  claim's words, not that the claim is true.
- Single process: paused plans and checkpoints live in memory.
