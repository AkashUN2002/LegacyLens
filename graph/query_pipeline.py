from langgraph.graph import StateGraph, START, END

from graph.state import GraphState
from agents.qa_agent import qa_agent


def build_query_pipeline():
    """
    Build and compile the P2 query pipeline.

    Flow (runs once per user question):

        START → qa_agent → END

    The QA agent internally handles intent classification, retrieval
    (vector search + $graphLookup), and answer synthesis. We keep it
    as a single node for now; a response_formatter node can be added
    later without changing callers.

    Unlike P1, this pipeline is stateless per invocation and needs no
    checkpointer — every question is independent. Conversation history
    is held by the UI layer (Streamlit session state), not the graph.

    Returns:
        A compiled LangGraph app. Invoke with:
            app.invoke({"question": ..., "db": ..., "repo_id": ...})
    """
    graph = StateGraph(GraphState)

    graph.add_node("qa_agent", qa_agent)

    graph.add_edge(START,      "qa_agent")
    graph.add_edge("qa_agent", END)

    return graph.compile()


# Build once at import time — the compiled graph is reusable across calls.
_QUERY_APP = None


def _get_query_app():
    global _QUERY_APP
    if _QUERY_APP is None:
        _QUERY_APP = build_query_pipeline()
    return _QUERY_APP


def run_query(question: str, db, repo_id: str, session_id: str = "default") -> dict:
    """
    Convenience runner for the query pipeline.

    Args:
        question:   the user's natural language question
        db:         live MongoDB database handle
        repo_id:    stable repo identifier (from the ingested metadata)
        session_id: conversation/session id for persistent memory

    Returns:
        A dict with the answer and supporting detail.
    """
    app = _get_query_app()

    initial_state: GraphState = {
        "question":   question,
        "db":         db,
        "repo_id":    repo_id,
        "session_id": session_id,
    }

    final_state = app.invoke(initial_state)

    return {
        "answer":         final_state.get("answer", ""),
        "cited_entities": final_state.get("cited_entities", []),
        "confidence":     final_state.get("confidence", "low"),
        "intent":         final_state.get("intent", ""),
        "target_entity":  final_state.get("target_entity"),
        "retrieved":      final_state.get("retrieved", []),
        "mcp_context":    final_state.get("mcp_context", ""),
        "mcp_tool_calls": final_state.get("mcp_tool_calls", []),
        "used_deterministic": final_state.get("used_deterministic", False),
    }