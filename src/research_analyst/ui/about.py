"""Static content for the in-app About tab and the example questions."""

SAMPLE_QUESTIONS = [
    "What are the main applications and risks of large language models in healthcare?",
    "How close are solid-state batteries to commercial electric vehicles, and what challenges remain?",
    "What is the current state of CRISPR-based gene therapies in clinical use?",
    "What does the evidence say about the effect of remote work on productivity?",
    "What progress has been made in quantum error correction and what limits scaling?",
]

ABOUT_MD = """
### What is this?
**An AI research assistant that turns a question into a cited report.** You type a topic; it breaks it into
sub-questions (you can edit them), searches the web and arXiv, reads the sources, writes each section from the
evidence it found, and assembles a report where every claim points to a numbered source.

### What can I use it for?
- A fast, **sourced first draft** on a topic you're new to (technology, health, science, policy, business)
- **Mapping a question** into sub-questions and spotting what is *not* well covered
- Collecting a **starting reading list** with links, including arXiv papers for technical topics

It is a research *aid*, not an authority. Always open the cited sources before relying on a claim.

### How it works
1. **Plan** — an LLM splits your question into 4–5 sub-questions; you review and edit them.
2. **Search** — each sub-question is searched in parallel (two phrasings; arXiv for technical topics).
   Results are ranked by relevance and a source-quality prior, and the best pages are fetched.
3. **Read** — facts are extracted from the best passages, each tied to its source.
4. **Write** — each section cites its facts inline; the final report's reference list is built by code
   from the sources actually cited, and any citation the model invented is removed.
5. **Check** — automated checks count uncited and weakly-supported sentences. An optional critic/reviser
   loop can be switched on (off by default: in testing it added citations but cost ~75% more and
   did not improve faithfulness).

### Built with
Python · LangGraph (parallel fan-out per sub-question, human approval step) · Gemini models only (a fast model for most steps, a stronger one for the final report; native function calling and
schema-constrained outputs, no fallback provider) · DuckDuckGo + arXiv · trafilatura + BM25 for page text · Streamlit ·
Langfuse tracing · pytest/mypy/ruff in CI. Source: github.com/charan-ravikumar/Multi-Agent-AI-Research-Analyst

### Good to know
- A run takes **about 3–6 minutes** (free-tier model limits).
- Answers are only as good as what the search finds. No paywalled or login-only pages, no PDFs
  (arXiv gives abstracts only).
- The citation counts are mechanical (markers present); the per-claim verdicts come from an LLM judge that reads only the gathered passages. Both are triage signals, not proof of truth.
"""
