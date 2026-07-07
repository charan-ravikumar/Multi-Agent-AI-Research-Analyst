"""
tools/eval_harness.py — Evaluation harness for completed research runs.

Reads a saved state (from tools/run_demo.py → logs/last_state.json) and
computes four quality metrics:

  relevance         per-section LLM score (1-10): does the section answer
                    its sub_question?
  faithfulness      fraction of sampled claims grounded in extracted_facts
                    (LLM-judged; paraphrase-aware)
  hallucination_rate 1 - faithfulness
  citation_coverage  fraction of sub_questions that have ≥1 citation

When both draft (pre-debate) and revised_draft (post-debate) are in state,
runs faithfulness on BOTH so you can see whether the debate loop actually
improved grounding.

WHY LLM-JUDGED AND NOT RAGAS / EMBEDDINGS
------------------------------------------
ragas: broken in this environment — import fails because the transformers
  package is corrupted (missing audio_spectrogram_transformer model file).
  Even if it were installed, ragas expects LangChain-wrapped LLMs and its
  own EvaluationDataset format; mapping ResearchState would add complexity
  with no gain.

sentence_transformers / embedding similarity: also broken (same transformers
  corruption).  No embedding capability exists in this codebase — the project
  deliberately uses Redis-only memory with no vector store.

LLM judgment is the right choice here anyway: a section can be semantically
  close to a sub_question (high cosine similarity) while still not answering
  it, or a paraphrased fact can be fully faithful while having near-zero
  token overlap.  Judgment requires understanding, not similarity.

WHY FAITHFULNESS MATTERS FOR THE DEBATE COMPARISON
---------------------------------------------------
The Synthesizer's revision prompt says: "If an objection says a claim lacks
corroboration, add an explicit caveat."  The key empirical question is whether
the revision ADDS CAVEATS (hedging without improving grounding) or REMOVES
CLAIMS (reducing hallucination) or ADDS GROUNDED FACTS (improving faithfulness).
The pre/post-debate faithfulness numbers give our first real evidence.

Usage
-----
    # Default: reads logs/last_state.json
    python tools/eval_harness.py

    # Explicit file
    python tools/eval_harness.py --state-file logs/last_state.json

    # Skip the LLM calls (rule-based metrics only) for a fast sanity check
    python tools/eval_harness.py --quick
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import time

import llm
from llm import ModelTier
from core.logger import get_logger

_log = get_logger("eval")
_EVAL_TIER = ModelTier.STRONG     # overridden via --tier flag

DEFAULT_STATE = ROOT / "logs" / "last_state.json"

# ── claim extraction (rule-based) ─────────────────────────────────────────────

_SKIP_PREFIXES = ("#", "*", "-", ">", "##", "###", "—", "–")
_MIN_CLAIM_LEN = 25


def extract_claims(text: str, max_claims: int = 12) -> list[str]:
    """
    Split text into candidate factual-claim sentences.

    Heuristics:
    - Split on sentence boundaries (. ! ?)
    - Discard very short fragments (<25 chars)
    - Discard markdown headers / bullet points
    - Discard questions (end with ?)
    - Limit to max_claims per call to keep token cost bounded
    """
    # Normalise markdown headers to plain text before splitting
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < _MIN_CLAIM_LEN:
            continue
        if any(s.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if s.endswith("?"):
            continue
        claims.append(s)
    return claims[:max_claims]


# ── LLM evaluation calls ──────────────────────────────────────────────────────

_RELEVANCE_SYSTEM = """\
You are an evaluator assessing whether research text directly answers its question.
For each (QUESTION, TEXT) pair, output an integer score 1–10:
  1  = completely off-topic, answers a different question
  5  = partially relevant, addresses related concepts but not the question directly
  10 = fully relevant, directly and completely answers the question

Respond with ONLY a valid JSON object — no markdown fences, no explanation, no prose:
{"scores": [<int>, <int>, ...]}
The list must have exactly as many integers as there are pairs.
Output nothing except the JSON object."""

_RELEVANCE_USER = """\
Evaluate these question-text pairs:

{pairs}

Respond with {n} integer scores in the JSON format shown."""


def _evaluate_relevance(pairs: list[tuple[str, str]]) -> list[int]:
    """
    Score each (sub_question, section_text) pair for direct relevance.
    Returns list of scores 1-10.  On parse failure, returns 5 for all.
    """
    if not pairs:
        return []

    formatted = "\n\n".join(
        f"Pair {i+1}:\n  QUESTION: {q}\n  TEXT: {t[:600]}"
        for i, (q, t) in enumerate(pairs)
    )
    prompt = _RELEVANCE_USER.format(pairs=formatted, n=len(pairs))

    for attempt in range(2):
        raw, _ = llm.generate(
            # system MUST be first element when passing a list — system= kwarg is ignored for lists
            [
                {"role": "system", "content": _RELEVANCE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tier=_EVAL_TIER,
            return_usage=True,
        )
        try:
            parsed = json.loads(_strip_fences(raw))
            scores = parsed["scores"]
            # Accept both int and float (LLM sometimes outputs 7.0)
            coerced = [int(round(s)) for s in scores]
            if len(coerced) == len(pairs):
                return coerced
        except Exception:
            pass  # retry

    # fallback
    return [5] * len(pairs)


_FAITHFULNESS_SYSTEM = """\
You are a fact-checker.  Given GROUND TRUTH FACTS extracted from source documents
and a list of CLAIMS to check, classify each claim into EXACTLY ONE of three
categories:

  "supported"   — The claim's core assertion can be traced to at least one ground
                  truth fact (exact wording not required; paraphrase is fine).

  "hedged"      — The claim EXPLICITLY signals its own uncertainty or lack of
                  corroboration using language such as: "not yet fully understood",
                  "requires further research" or "investigation", "based on limited
                  available evidence", "some initial research suggests ... though
                  corroboration is lacking", "more research is needed", "remains an
                  area of ongoing research", "represents a gap in the current
                  dataset", "not independently verified", "preliminary findings
                  indicate", or similar epistemic qualifiers.  A hedged claim
                  honestly discloses its own uncertainty — this is the CORRECT
                  behavior for a claim lacking full corroboration, not a failure.

  "unsupported" — The claim makes a specific, confident factual assertion (with NO
                  hedging language) that does NOT appear in any ground truth fact.
                  Examples of unsupported claims: fabricated statistics ("reduces
                  costs by 25%"), invented journal names, named institutions not
                  mentioned in the facts, specific study results not in the facts,
                  and confident causal claims with no factual basis.  This is the
                  ONLY category that represents genuine hallucination.

Respond with ONLY a valid JSON object — no markdown fences, no explanation, no prose:
{"verdicts": ["supported", "hedged", "unsupported", ...]}
The list must have exactly as many strings as there are claims.
Output nothing except the JSON object."""

_FAITHFULNESS_USER = """\
GROUND TRUTH FACTS ({n_facts} total):
{facts_block}

CLAIMS TO CHECK ({n_claims}):
{claims_block}

Classify each claim as "supported", "hedged", or "unsupported".
Respond with {n_claims} strings in the JSON format shown."""


_VALID_VERDICTS = frozenset({"supported", "hedged", "unsupported"})


def _evaluate_faithfulness(
    claims: list[str],
    facts: list[dict],
    batch_size: int = 10,
) -> list[str | None]:
    """
    Classify each claim as 'supported', 'hedged', or 'unsupported'.

    supported   — traceable to an extracted fact (paraphrase OK)
    hedged      — claim explicitly signals its own uncertainty/gap; correct
                  behavior, NOT counted as hallucination
    unsupported — confident factual assertion not in extracted facts; the
                  only category that constitutes genuine hallucination

    Processes in batches.  Returns list of verdict strings (or None on
    parse failure).
    """
    if not claims:
        return []

    facts_text = "\n".join(
        f"[F{i+1}] {f['content']}"
        for i, f in enumerate(facts[:30])
    )

    results: list[str | None] = []
    for i in range(0, len(claims), batch_size):
        batch = claims[i : i + batch_size]
        claims_block = "\n".join(f"[C{j+1}] {c}" for j, c in enumerate(batch))

        prompt = _FAITHFULNESS_USER.format(
            n_facts=min(len(facts), 30),
            facts_block=facts_text,
            n_claims=len(batch),
            claims_block=claims_block,
        )

        parsed_ok = False
        time.sleep(2)   # brief pause between batches to stay within TPM
        for attempt in range(2):
            raw, _ = llm.generate(
                [
                    {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tier=_EVAL_TIER,
                return_usage=True,
            )
            try:
                parsed = json.loads(_strip_fences(raw))
                verdicts = parsed["verdicts"]
                # Normalise to lower-case and validate
                coerced = [str(v).lower().strip() for v in verdicts]
                if (
                    len(coerced) == len(batch)
                    and all(v in _VALID_VERDICTS for v in coerced)
                ):
                    results.extend(coerced)
                    parsed_ok = True
                    break
            except Exception:
                pass

        if not parsed_ok:
            results.extend([None] * len(batch))

    return results


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ── citation quality (rule-based) ─────────────────────────────────────────────

def evaluate_citation_coverage(
    sub_questions: list[str],
    citations: list[dict],
) -> dict:
    """
    For each sub_question, count citations that are linked to it.
    A citation is linked if its 'snippet' mentions the sub_question keywords
    OR if no sub_question field is present (assume it covers all).
    Also counts total citations and flags any with missing/empty URLs.
    """
    # In this system, citations don't carry a sub_question field explicitly —
    # they're accumulated per-branch.  Check for empty/invalid URLs.
    total = len(citations)
    empty_url = sum(1 for c in citations if not c.get("url", "").strip())
    valid = total - empty_url

    # Check coverage per sub_question using citation snippets as proxy.
    # This is approximate — a citation with a snippet mentioning key terms
    # from a sub_question counts as covering it.
    covered_sqs: list[bool] = []
    for sq in sub_questions:
        sq_keywords = set(re.findall(r"\b[a-z]{4,}\b", sq.lower())) - {
            "what", "how", "does", "this", "that", "with", "from", "such",
            "large", "each", "than", "have",
        }
        found = any(
            any(kw in c.get("snippet", "").lower() for kw in sq_keywords)
            for c in citations
        )
        covered_sqs.append(found)

    return {
        "total_citations": total,
        "valid_url_count": valid,
        "empty_url_count": empty_url,
        "sub_questions_with_citation_coverage": sum(covered_sqs),
        "sub_questions_total": len(sub_questions),
        "coverage_fraction": sum(covered_sqs) / max(len(sub_questions), 1),
        "per_sq_coverage": list(zip(sub_questions, covered_sqs)),
    }


# ── main evaluation runner ────────────────────────────────────────────────────

def run_evaluation(state: dict, quick: bool = False) -> dict:
    """
    Run all metrics against a saved state dict.  Returns a structured
    results dict and logs everything to agent_traces.jsonl.
    """
    session_id: str = state.get("session_id", "unknown")
    sub_questions: list[str] = state.get("sub_questions", [])
    extracted_facts: list[dict] = state.get("extracted_facts", [])
    citations: list[dict] = state.get("citations", [])
    draft: list[str] = state.get("draft", [])
    revised_draft: list[str] = state.get("revised_draft") or []
    final_report: str = state.get("final_report", "")
    debate_fired: bool = bool(revised_draft) and revised_draft != draft

    print(f"\n{'='*70}")
    print(f"  EVALUATION HARNESS")
    print(f"  session_id : {session_id}")
    print(f"  sub_questions : {len(sub_questions)}")
    print(f"  extracted_facts : {len(extracted_facts)}")
    print(f"  citations : {len(citations)}")
    print(f"  draft sections (pre-debate) : {len(draft)}")
    print(f"  revised_draft sections (post-debate) : {len(revised_draft)}")
    print(f"  debate fired : {debate_fired}")
    print(f"  quick mode (no LLM calls) : {quick}")
    print(f"{'='*70}\n")

    results: dict[str, Any] = {
        "session_id": session_id,
        "debate_fired": debate_fired,
    }

    # ── 1. Citation coverage (rule-based — always runs) ───────────────────────
    print("[ 1/4 ] Citation coverage (rule-based)...")
    cit_result = evaluate_citation_coverage(sub_questions, citations)
    results["citation_coverage"] = cit_result
    _print_section("Citation Coverage", cit_result)

    if quick:
        print("\n[quick mode] Skipping LLM evaluation metrics.")
        return results

    # ── 2. Relevance (LLM-judged) ──────────────────────────────────────────────
    print("\n[ 2/4 ] Relevance: sub_question ↔ draft section (LLM-judged)...")
    pre_relevance_scores = _eval_relevance_for_draft(sub_questions, draft, "pre-debate")
    results["relevance_pre_debate"] = {
        "scores": pre_relevance_scores,
        "mean": round(sum(pre_relevance_scores) / max(len(pre_relevance_scores), 1), 2),
        "per_sq": list(zip(sub_questions, pre_relevance_scores)),
    }

    post_relevance_scores: list[int] = []
    if debate_fired:
        print("  [post-debate] scoring revised_draft relevance...")
        post_relevance_scores = _eval_relevance_for_draft(
            sub_questions, revised_draft, "post-debate"
        )
        results["relevance_post_debate"] = {
            "scores": post_relevance_scores,
            "mean": round(
                sum(post_relevance_scores) / max(len(post_relevance_scores), 1), 2
            ),
            "per_sq": list(zip(sub_questions, post_relevance_scores)),
        }

    # ── 3. Faithfulness / hallucination rate (LLM-judged) ────────────────────
    print("\n[ 3/4 ] Faithfulness (LLM-judged, pre-debate draft)...")
    pre_faith = _eval_faithfulness_for_draft(draft, extracted_facts, "pre-debate")
    results["faithfulness_pre_debate"] = pre_faith
    _print_faithfulness("Pre-debate draft", pre_faith)

    post_faith: dict = {}
    if debate_fired:
        print("\n[ 3b/4 ] Faithfulness (LLM-judged, post-debate revised_draft)...")
        post_faith = _eval_faithfulness_for_draft(
            revised_draft, extracted_facts, "post-debate"
        )
        results["faithfulness_post_debate"] = post_faith
        _print_faithfulness("Post-debate revised_draft", post_faith)

    # ── 4. Summary comparison ─────────────────────────────────────────────────
    print("\n[ 4/4 ] Summary...")
    results["summary"] = _build_summary(
        pre_relevance_scores, post_relevance_scores, pre_faith, post_faith, cit_result
    )
    _print_summary(results["summary"], debate_fired, pre_faith, post_faith)

    # ── Log to agent_traces.jsonl ─────────────────────────────────────────────
    _log.info(
        "Evaluation complete",
        step="eval_complete",
        session_id=session_id,
        relevance_pre=results.get("relevance_pre_debate", {}).get("mean"),
        relevance_post=results.get("relevance_post_debate", {}).get("mean"),
        faithfulness_pre=pre_faith.get("faithfulness"),
        faithfulness_post=post_faith.get("faithfulness") if post_faith else None,
        disclosure_pre=pre_faith.get("disclosure_rate"),
        disclosure_post=post_faith.get("disclosure_rate") if post_faith else None,
        hallucination_pre=pre_faith.get("hallucination_rate"),
        hallucination_post=post_faith.get("hallucination_rate") if post_faith else None,
        citation_coverage=cit_result.get("coverage_fraction"),
    )

    return results


def _eval_relevance_for_draft(
    sub_questions: list[str], draft: list[str], label: str
) -> list[int]:
    """Batch-score all (sub_question, section) pairs for relevance."""
    if not draft or not sub_questions:
        return []
    pairs = list(zip(sub_questions, draft))
    scores = _evaluate_relevance(pairs)
    for sq, score in zip(sub_questions, scores):
        print(f"  [{label}] score={score}/10  sq={sq[:60]}")
    return scores


def _eval_faithfulness_for_draft(
    draft: list[str], extracted_facts: list[dict], label: str
) -> dict:
    """Run faithfulness checks across all draft sections."""
    all_claims: list[str] = []
    for section in draft:
        claims = extract_claims(section, max_claims=8)
        all_claims.extend(claims)

    print(f"  [{label}] {len(all_claims)} claims extracted from {len(draft)} sections")

    if not all_claims:
        return {
            "claims_checked": 0,
            "supported": 0,
            "hedged": 0,
            "unsupported": 0,
            "parse_failures": 0,
            "faithfulness": None,
            "disclosure_rate": None,
            "hallucination_rate": None,
            "claims": [],
        }

    verdicts = _evaluate_faithfulness(all_claims, extracted_facts)

    n_supported   = sum(1 for v in verdicts if v == "supported")
    n_hedged      = sum(1 for v in verdicts if v == "hedged")
    n_unsupported = sum(1 for v in verdicts if v == "unsupported")
    n_failures    = sum(1 for v in verdicts if v is None)
    n_checkable   = n_supported + n_hedged + n_unsupported

    faithfulness      = round(n_supported   / max(n_checkable, 1), 3)
    disclosure_rate   = round(n_hedged      / max(n_checkable, 1), 3)
    hallucination_rate = round(n_unsupported / max(n_checkable, 1), 3)

    claim_details = [
        {"claim": c, "verdict": v}
        for c, v in zip(all_claims, verdicts)
    ]

    _FLAG = {"supported": "✓", "hedged": "~", "unsupported": "✗", None: "?"}
    for c, v in zip(all_claims, verdicts):
        print(f"    [{_FLAG.get(v, '?')}] {c[:80]}")

    print(
        f"  [{label}] supported={n_supported}  hedged={n_hedged}  "
        f"unsupported={n_unsupported}  (of {n_checkable} checkable)\n"
        f"  [{label}] faithfulness={faithfulness:.1%}  "
        f"disclosure={disclosure_rate:.1%}  "
        f"hallucination={hallucination_rate:.1%}"
    )

    return {
        "claims_checked": len(all_claims),
        "supported": n_supported,
        "hedged": n_hedged,
        "unsupported": n_unsupported,
        "parse_failures": n_failures,
        "faithfulness": faithfulness,
        "disclosure_rate": disclosure_rate,
        "hallucination_rate": hallucination_rate,
        "claims": claim_details,
    }


def _build_summary(
    pre_rel: list[int],
    post_rel: list[int],
    pre_faith: dict,
    post_faith: dict,
    cit: dict,
) -> dict:
    return {
        "relevance_pre": round(sum(pre_rel) / max(len(pre_rel), 1), 2) if pre_rel else None,
        "relevance_post": round(sum(post_rel) / max(len(post_rel), 1), 2) if post_rel else None,
        "faithfulness_pre": pre_faith.get("faithfulness"),
        "faithfulness_post": post_faith.get("faithfulness"),
        "disclosure_pre": pre_faith.get("disclosure_rate"),
        "disclosure_post": post_faith.get("disclosure_rate"),
        "hallucination_pre": pre_faith.get("hallucination_rate"),
        "hallucination_post": post_faith.get("hallucination_rate"),
        "citation_coverage": cit.get("coverage_fraction"),
    }


def _print_section(label: str, data: dict) -> None:
    print(f"\n  {label}:")
    for k, v in data.items():
        if k == "per_sq_coverage":
            for sq, covered in v:
                mark = "✓" if covered else "✗"
                print(f"    [{mark}] {sq[:70]}")
        else:
            print(f"    {k}: {v}")


def _print_faithfulness(label: str, data: dict) -> None:
    f  = data.get("faithfulness")
    d  = data.get("disclosure_rate")
    h  = data.get("hallucination_rate")
    n  = data.get("claims_checked", 0)
    s  = data.get("supported", 0)
    hd = data.get("hedged", 0)
    u  = data.get("unsupported", 0)
    pf = data.get("parse_failures", 0)
    print(f"\n  {label}:")
    print(f"    claims checked   : {n}")
    print(f"    [✓] supported    : {s}")
    print(f"    [~] hedged       : {hd}   (honest disclosure — NOT hallucination)")
    print(f"    [✗] unsupported  : {u}   (genuine hallucination)")
    if pf:
        print(f"    [?] parse fail  : {pf}")
    print(f"    faithfulness     : {f:.1%}" if f is not None else "    faithfulness     : N/A")
    print(f"    disclosure rate  : {d:.1%}" if d is not None else "    disclosure rate  : N/A")
    print(f"    hallucination    : {h:.1%}" if h is not None else "    hallucination    : N/A")


def _print_summary(s: dict, debate_fired: bool, pre_faith: dict, post_faith: dict) -> None:
    W = 70
    print(f"\n{'='*W}")
    print(f"  SCORE COMPARISON {'(debate fired)' if debate_fired else '(no debate)'}")
    print(f"{'='*W}")
    print(f"  {'Metric':<30} {'Pre-debate':>12} {'Post-debate':>12} {'Delta':>8}")
    print(f"  {'-'*62}")

    def _fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.1%}" if v <= 1 else f"{v:.1f}"
        return str(v)

    def _delta(pre, post):
        if pre is None or post is None:
            return "—"
        d = post - pre
        sign = "+" if d > 0 else ""
        if isinstance(pre, float) and pre <= 1:
            return f"{sign}{d:.1%}"
        return f"{sign}{d:.1f}"

    rows = [
        ("Relevance (mean 1-10)", s["relevance_pre"], s["relevance_post"]),
        ("Faithfulness (supported/total)", s["faithfulness_pre"], s["faithfulness_post"]),
        ("Disclosure rate (hedged/total)", s["disclosure_pre"], s["disclosure_post"]),
        ("Hallucination (unsupported/total)", s["hallucination_pre"], s["hallucination_post"]),
        ("Citation coverage", s["citation_coverage"], None),
    ]
    for label, pre, post in rows:
        print(
            f"  {label:<30} {_fmt(pre):>12} {_fmt(post):>12} {_delta(pre, post):>8}"
        )
    print(f"{'='*W}")

    # Honest interpretation — uses hallucination_rate (unsupported only),
    # not faithfulness, as the primary grounding-quality signal.
    print("\n  INTERPRETATION:")
    pre_h  = s.get("hallucination_pre")
    post_h = s.get("hallucination_post")
    pre_d  = s.get("disclosure_pre")
    post_d = s.get("disclosure_post")
    pre_f  = s.get("faithfulness_pre")
    post_f = s.get("faithfulness_post")

    if pre_h is None:
        print("  No faithfulness data (quick mode or no claims extracted).")
    elif post_h is None:
        print(f"  Debate did not fire.  Pre-debate hallucination: {pre_h:.1%}")
    else:
        h_delta = post_h - pre_h
        d_delta = (post_d - pre_d) if (post_d is not None and pre_d is not None) else None

        print(f"  Hallucination (unsupported/total): {pre_h:.1%} → {post_h:.1%}  (Δ{h_delta:+.1%})")
        if d_delta is not None:
            print(f"  Disclosure   (hedged/total)     : {pre_d:.1%} → {post_d:.1%}  (Δ{d_delta:+.1%})")
        if pre_f is not None and post_f is not None:
            print(f"  Faithfulness (supported/total)  : {pre_f:.1%} → {post_f:.1%}")
        print()

        if post_h < 0.05:
            print("  ✔  HALLUCINATION ESSENTIALLY ELIMINATED after debate revision.")
            print(f"  Remaining {post_h:.0%} is within noise tolerance for LLM judgment.")
        elif post_h < pre_h - 0.10:
            print(f"  ✔  Hallucination REDUCED substantially ({pre_h:.1%} → {post_h:.1%}).")
        elif abs(post_h - pre_h) < 0.05:
            print("  ○  Hallucination FLAT — debate loop did not change real fabrication rate.")
        else:
            print(f"  ✘  Hallucination INCREASED ({pre_h:.1%} → {post_h:.1%}).")
            print("  The revision added fabricated content beyond the original draft.")

        if d_delta is not None and post_d > pre_d + 0.05:
            print(f"  ✔  Disclosure INCREASED ({pre_d:.1%} → {post_d:.1%}): the revised")
            print("  draft added honest epistemic hedges where facts were missing —")
            print("  the intended behavior of the fixed revision prompt.")
        elif d_delta is not None and post_d < 0.05:
            print(f"  ○  Disclosure LOW ({post_d:.1%}): few hedges added.")

    pre_rel = s["relevance_pre"]
    post_rel = s["relevance_post"]
    if pre_rel and post_rel:
        r_delta = post_rel - pre_rel
        if abs(r_delta) < 0.5:
            print(f"\n  Relevance: {pre_rel:.1f} → {post_rel:.1f} — essentially flat.")
        elif r_delta < 0:
            print(f"\n  Relevance DECLINED: {pre_rel:.1f} → {post_rel:.1f}.")
        else:
            print(f"\n  Relevance IMPROVED: {pre_rel:.1f} → {post_rel:.1f}.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-file", default=str(DEFAULT_STATE), help=f"Path to state JSON (default: {DEFAULT_STATE})")
    parser.add_argument("--quick", action="store_true", help="Skip LLM calls; run rule-based metrics only")
    parser.add_argument("--out", help="Save results JSON to this path")
    parser.add_argument("--tier", choices=["strong", "fast"], default="strong",
                        help="LLM tier for evaluation (default: strong). Use 'fast' when STRONG-tier daily budget is exhausted.")
    args = parser.parse_args()

    global _EVAL_TIER
    _EVAL_TIER = ModelTier.FAST if args.tier == "fast" else ModelTier.STRONG

    state_path = Path(args.state_file)
    if not state_path.exists():
        print(f"State file not found: {state_path}", file=sys.stderr)
        print("Run tools/run_demo.py first to generate a state file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading state from {state_path} ({state_path.stat().st_size // 1024} KB)...")
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    results = run_evaluation(state, quick=args.quick)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
