from langgraph.graph import StateGraph, START, END

from graph.state import GraphState
from agents.parser_agent import parser_agent
from agents.graph_builder_agent import graph_builder_agent
from agents.risk_analyst_agent import risk_analyst_agent


def build_ingestion_pipeline(checkpointer=None):
    """
    Build and compile the P1 ingestion pipeline.

    Flow (linear, runs once per repo):

        START → parser → graph_builder → risk_analyst → END

    Args:
        checkpointer: optional LangGraph checkpointer (e.g. MongoDBSaver).
                      If provided, the pipeline can resume from the last
                      completed node if it crashes mid-run.

    Returns:
        A compiled LangGraph app. Invoke with:
            app.invoke({"repo_path": ..., "language": ..., "db": ...})
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("parser",        parser_agent)
    graph.add_node("graph_builder", graph_builder_agent)
    graph.add_node("risk_analyst",  risk_analyst_agent)

    # Wire the linear flow
    graph.add_edge(START,           "parser")
    graph.add_edge("parser",        "graph_builder")
    graph.add_edge("graph_builder", "risk_analyst")
    graph.add_edge("risk_analyst",  END)

    return graph.compile(checkpointer=checkpointer)


def run_ingestion(repo_path: str, language: str, db, checkpointer=None,
                  thread_id: str = "ingestion") -> GraphState:
    """
    Convenience runner for the ingestion pipeline.

    Args:
        repo_path:    path to the repo root
        language:     "python" or "java"
        db:           live MongoDB database handle
        checkpointer: optional MongoDBSaver
        thread_id:    checkpoint thread identifier

    Returns:
        The final GraphState after ingestion completes.
    """
    app = build_ingestion_pipeline(checkpointer=checkpointer)

    initial_state: GraphState = {
        "repo_path": repo_path,
        "language":  language,
        "db":        db,
    }

    config = {"configurable": {"thread_id": thread_id}} if checkpointer else {}

    print(f"[ingestion] starting pipeline for {repo_path!r} ({language})")
    final_state = app.invoke(initial_state, config=config)
    print(
        f"[ingestion] complete — "
        f"{len(final_state.get('entities', []))} entities, "
        f"{len(final_state.get('edges', []))} edges, "
        f"{final_state.get('high_risk_count', 0)} high-risk"
    )

    return final_state