"""
llm — public LLM interface for agents

Agents import ONLY from this package:

    import llm
    response = llm.generate("Explain AI")

    from llm import generate, ModelTier
    response = generate("Summarise this", tier=ModelTier.FAST)

Which provider(s) answer is an implementation detail hidden inside this
package.  Agents must never import from llm.groq_client, llm.gemini_client,
or llm.client directly.
"""
from llm.client import generate, ModelTier

__all__ = ["generate", "ModelTier"]
