"""Offline test helpers: scripted fake LLM, fake search, initial state."""
from __future__ import annotations

import json
import uuid

from research_analyst.models import ResearchState
from research_analyst.models.search_result import SearchResult

# ── fake LLM ──────────────────────────────────────────────────────────────────

_ROLES = [
    ("research planning assistant", "planner"),
    ("fact-extraction assistant", "reader"),
    ("research synthesis assistant", "synth"),
    ("critical reviewer", "critic"),
    ("research revision assistant", "revise"),
    ("research writer", "writer"),
    ("You write web search queries", "rewriter"),
    ("research search agent", "react"),
    ("strict fact-checking judge", "judge"),
]

_WRITER_OK = (
    "## Executive Summary\nx\n## Key Findings\n- y\n## Detailed Analysis\nz\n"
    "## Contradictions and Conflicting Evidence\nNone flagged.\n"
    "## Knowledge Gaps and Limitations\n- Some limits are noted here.\n## References\n[1] t"
)


class FakeLLM:
    """Scripted stand-in for llm.generate; dispatches on the agent's system prompt."""

    def __init__(self, n_sub_questions: int = 4, overrides: dict | None = None):
        self.n = n_sub_questions
        self.overrides = overrides or {}
        self.calls: dict[str, int] = {}
        self.prompts: list[tuple[str, list]] = []
        self.kwargs: list[tuple[str, dict]] = []

    @staticmethod
    def role_of(messages) -> str:
        system = messages[0]["content"] if isinstance(messages, list) else ""
        for key, role in _ROLES:
            if key in system:
                return role
        return "?"

    def _default(self, role: str) -> str:
        if role == "planner":
            return json.dumps({
                "query": "q",
                "sub_questions": [f"What is aspect {i} of the topic?" for i in range(self.n)],
                "depth": "standard", "strategy_notes": "n",
            })
        return {
            "reader": json.dumps({"facts": [{"source": 1, "content": "A fact.", "confidence": 0.9}]}),
            "synth": json.dumps({"synthesis": "A synthesis paragraph.", "contradictions": [], "gaps": []}),
            "critic": json.dumps({"objections": []}),
            "revise": json.dumps({"revised_sections": ["revised section"]}),
            "writer": _WRITER_OK,
            "rewriter": json.dumps({"queries": ["rewritten keyword query about the topic"]}),
            "react": json.dumps({"thought": "enough", "action": "finish", "input": ""}),
            "judge": json.dumps({"verdicts": []}),
        }.get(role, "{}")

    def __call__(self, prompt, *, tier=None, system=None, temperature=None, max_tokens=None,
                 return_usage=False, json_mode=False, schema=None, model=None, on_text=None):
        role = self.role_of(prompt)
        self.calls[role] = self.calls.get(role, 0) + 1
        self.prompts.append((role, prompt))
        self.kwargs.append((role, {"json_mode": json_mode, "max_tokens": max_tokens, "schema": schema, "model": model, "tier": tier}))
        fn = self.overrides.get(role)
        out = fn(prompt, self.calls[role]) if fn else self._default(role)
        if on_text is not None:
            on_text(out)
        usage = {"provider": "fake", "model": "fake", "prompt_tokens": 1, "completion_tokens": 1,
                 "total_tokens": 2, "latency_s": 0.0, "finish_reason": "stop"}
        return (out, usage) if return_usage else out

    @property
    def total(self) -> int:
        return sum(self.calls.values())


def make_search(n_results: int = 3, snippet: str = "A snippet.", calls: list | None = None):
    async def _search(query, *, sub_question="", max_results=5):
        if calls is not None:
            calls.append(query)
        return [
            SearchResult(sub_question=sub_question or query, url=f"https://example.com/{abs(hash(query)) % 9999}/{i}",
                         title=f"T{i}", snippet=snippet)
            for i in range(n_results)
        ]
    return _search


def initial_state(auto_approve: bool = True, query: str = "What is X?") -> ResearchState:
    return ResearchState(
        session_id=str(uuid.uuid4()), query=query, research_plan=None, sub_questions=[],
        search_results=[], react_traces=[], failed_sub_questions=[], extracted_facts=[], citations=[],
        contradictions=[], unresolved_gaps=[], gap_items=[], draft=[], final_report="", iteration=0,
        routing_history=[], auto_approve_plan=auto_approve,
    )




def make_pdf(pages: list[str]) -> bytes:
    """A minimal but valid text PDF (Helvetica, one line of text per page) for extraction tests."""
    objs: list[bytes] = []

    def add(body: str) -> int:
        objs.append(body.encode("latin-1"))
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")                                   # 1
    kids_placeholder = len(objs) + 1
    add("")                                                                     # 2 (filled below)
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")               # 3
    page_ids = []
    for text in pages:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        content_id = len(objs) + 2
        page_ids.append(add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                            f"/Resources << /Font << /F1 3 0 R >> >> >>"))
        add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs[kids_placeholder - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(page_ids)} >>").encode("latin-1")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
