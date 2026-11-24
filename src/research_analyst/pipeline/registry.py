"""Agent instances shared by the graph nodes (one per process; each holds only config + a logger)."""
from __future__ import annotations

from research_analyst.agents.critic import CriticAgent
from research_analyst.agents.planner import PlannerAgent
from research_analyst.agents.reader import ReaderAgent
from research_analyst.agents.searcher import SearcherAgent
from research_analyst.agents.synthesizer import SynthesizerAgent
from research_analyst.agents.writer import WriterAgent

planner = PlannerAgent()
searcher = SearcherAgent()
reader = ReaderAgent()
synthesizer = SynthesizerAgent()
critic = CriticAgent()
writer = WriterAgent()
