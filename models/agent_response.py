from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class AgentResponse(BaseModel):
    """
    Wraps the raw text output from llm.generate() with provenance metadata.
    Agents build one of these from the (text, usage) tuple and then extract
    the structured payload they need from `content`.

    Usage inside an agent:

        text, usage = llm.generate("...", return_usage=True)
        response = AgentResponse(
            content=text,
            agent="planner",
            step="generate_sub_questions",
            session_id=state["session_id"],
            iteration=state["iteration"],
            latency_s=usage["latency_s"],
            tokens=usage,
        )
        log.log_call(response.step, session_id=response.session_id,
                     iteration=response.iteration, latency_s=response.latency_s,
                     tokens=response.tokens)
    """

    content: str
    agent: str
    step: str
    session_id: str
    iteration: int = 0
    latency_s: Optional[float] = None
    tokens: Optional[Dict[str, Any]] = None   # provider-agnostic usage dict
