"""
pipeline/graph.py — assembles the LangGraph StateGraph.

    START -> planner -> plan_approval (human, interrupt)
          -> [fan_out: one Send per sub-question] -> process_sub_question (parallel, failure-isolated)
          -> supervisor -> debate (optional review loop) -> writer -> grounding_judge -> memory_store -> END

Node implementations live in pipeline/nodes/; agent instances in pipeline/registry.py.
Every caller must pass config={"configurable": {"thread_id": ...}} (the checkpointer is required for
interrupt()); automated callers set auto_approve_plan=True in the initial state.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_analyst.models import ResearchState
from research_analyst.pipeline import registry
from research_analyst.pipeline.nodes.branch import fan_out, process_sub_question
from research_analyst.pipeline.nodes.debate import debate_loop_node
from research_analyst.pipeline.nodes.grounding import grounding_judge_node
from research_analyst.pipeline.nodes.memory import memory_node
from research_analyst.pipeline.nodes.plan import plan_approval_node
from research_analyst.pipeline.nodes.supervisor import supervisor_node


def build_graph(checkpointer: Any = None) -> Any:   # returns a CompiledStateGraph
    """Construct and compile the research pipeline.

    Args:
        checkpointer: LangGraph checkpointer for persistence and interrupt
            support.  Pass MemorySaver() (or any other checkpointer) to enable
            interrupt()-based human-in-the-loop checkpoints.  Callers MUST
            then include config={"configurable": {"thread_id": ...}} in every
            invoke() / ainvoke() call.  Pass None to compile without a
            checkpointer (interrupt() will not work but no thread_id needed).
    """
    graph = StateGraph(ResearchState)

    # nodes
    # LangGraph types node inputs as the state TypedDict; our nodes accept partial-state mappings, hence the ignores.
    graph.add_node("planner",              registry.planner)  # type: ignore[type-var]
    graph.add_node("plan_approval",        plan_approval_node)
    graph.add_node("process_sub_question", process_sub_question)  # type: ignore[type-var]
    graph.add_node("supervisor",           supervisor_node)
    graph.add_node("debate",               debate_loop_node)
    graph.add_node("writer",               registry.writer)  # type: ignore[type-var]
    graph.add_node("grounding_judge",      grounding_judge_node)
    graph.add_node("memory_store",          memory_node)

    # edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "plan_approval")               # -> human checkpoint
    graph.add_conditional_edges("plan_approval", fan_out)   # -> fan-out
    graph.add_edge("process_sub_question", "supervisor")     # fan-in
    graph.add_edge("supervisor", "debate")                   # -> debate loop
    graph.add_edge("debate", "writer")                       # -> writer
    graph.add_edge("writer", "grounding_judge")      # -> LLM check of every cited claim (failure-isolated)
    graph.add_edge("grounding_judge", "memory_store")  # -> persist report
    graph.add_edge("memory_store", END)

    return graph.compile(checkpointer=checkpointer)


# Module-level compiled app (import-ready), with an in-process checkpointer so interrupt() works.
app = build_graph(checkpointer=MemorySaver())
