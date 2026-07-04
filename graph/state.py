from typing import TypedDict, Any, Optional


class GraphState(TypedDict, total=False):
    """
    Shared state object passed between all LangGraph nodes.

    `total=False` means no field is required at construction time —
    each agent reads the fields it needs and writes the fields it owns.
    This is what lets the same state type serve both pipelines.

    --------------------------------------------------------------------
    Field ownership (who writes what)
    --------------------------------------------------------------------
    """

    # ---- Injected by the caller / pipeline runner ----
    repo_path: str          # path to the repo root (ingestion only)
    language: str           # "python" | "java"
    db: Any                 # live MongoDB database handle (injected, not serialised)
    repo_id: str            # stable repo identifier (set by parser_agent)

    # ---- Written by parser_agent (P1, step 1) ----
    entities: list[dict]        # all extracted code entities
    parse_errors: list[str]     # files that failed to parse

    # ---- Written by graph_builder_agent (P1, step 2) ----
    edges: list[dict]           # resolved dependency edges
    graph_ready: bool           # True once edges are persisted

    # ---- Written by risk_analyst_agent (P1, step 3) ----
    risk_scores: dict[str, dict]    # entity_id -> score record
    high_risk_count: int            # number of high-risk entities
    embeddings_ready: bool          # True once vectors are written
    ingestion_done: bool            # True marks P1 complete

    # ---- Written by qa_agent (P2) ----
    question: str                   # the user's natural language question
    session_id: str                 # conversation/session id for memory
    intent: str                     # classified intent
    target_entity: Optional[str]    # named entity in the question, if any
    retrieved: list[dict]           # entities used as answer context
    answer: str                     # the cited natural language answer
    cited_entities: list[str]       # entity_ids the answer is based on
    confidence: str                 # high | medium | low
    mcp_context: str                # free-text context returned by the MCP agent
    mcp_tool_calls: list[dict]      # MCP tools invoked: name, args, result
    used_deterministic: bool        # whether a deterministic path served retrieval