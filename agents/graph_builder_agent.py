import os
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock

from db.schema import ENTITIES, EDGES, METADATA


# ---------------------------------------------------------------------------
# Bedrock LLM setup
# ---------------------------------------------------------------------------

# Bedrock requires a cross-region INFERENCE PROFILE id (e.g. "us.anthropic...")
# for on-demand calls — the bare model id is rejected. Configurable via .env so
# you can swap models without editing code. The default targets the US region.
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"


def _get_llm() -> ChatBedrock:
    """
    Returns a ChatBedrock instance.

    The model id is read from BEDROCK_MODEL_ID in the environment, falling
    back to a current cross-region inference-profile id. Region and
    credentials come from the standard AWS env vars / credential chain.
    """
    model_id = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)
    region   = os.environ.get("AWS_REGION", "us-east-1")
    return ChatBedrock(
        model_id=model_id,
        region_name=region,
    )


# ---------------------------------------------------------------------------
# Pydantic model — structured output from Claude
# ---------------------------------------------------------------------------

class EntitySummary(BaseModel):
    """
    Structured summary Claude generates for each code entity.
    Used with .with_structured_output() so the output is always
    a typed Pydantic object — never free text that needs parsing.
    """
    purpose: str = Field(
        description="One sentence: what this function or class does."
    )
    dependencies_note: str = Field(
        description=(
            "One sentence about its dependencies — e.g. "
            "'Calls TokenService and AuditLogger to process payments.'"
        )
    )
    modernisation_note: str = Field(
        description=(
            "One sentence flagging anything that would make this hard "
            "to modernise — e.g. tight coupling, missing error handling, "
            "no tests, or business logic buried in utility methods."
        )
    )


# ---------------------------------------------------------------------------
# Edge resolution
# ---------------------------------------------------------------------------

def _build_entity_index(entities: list[dict]) -> dict[str, dict]:
    """
    Build a lookup dict: entity_id -> entity document.
    """
    return {e["entity_id"]: e for e in entities}


def _entity_name(entity_id: str) -> str:
    """
    Extract the matchable name from an entity_id.

    entity_id format is "path::name" or "path::name::line".
    We want the name segment (the part between the first '::' and the
    optional trailing '::line'). For example:
        "src/p.py::PaymentProcessor.process_card::12" -> "PaymentProcessor.process_card"
    """
    parts = entity_id.split("::")
    if len(parts) >= 2:
        return parts[1]
    return entity_id


def _build_name_index(entities: list[dict]) -> dict[str, list[dict]]:
    """
    Build a lookup: name -> list of entities with that name.
    Calls are resolved against this, not against the line-suffixed entity_id.

    A name can map to several entities (overloads, redefinitions), so the
    value is a list. We also index the last dotted segment so that a call
    captured as a bare method name ("tokenize") still matches an entity
    stored as "TokenService.tokenize".
    """
    idx: dict[str, list[dict]] = {}
    for e in entities:
        name = _entity_name(e["entity_id"])
        idx.setdefault(name, []).append(e)
        # Also index the trailing segment, e.g. "tokenize" from "Class.tokenize"
        short = name.split(".")[-1]
        if short != name:
            idx.setdefault(short, []).append(e)
    return idx


def _resolve_edges(entities: list[dict], name_index: dict[str, list[dict]]) -> list[dict]:
    """
    For every entity, iterate its calls[] array and match each called name
    against the name index. Produces deduplicated edge documents:
        { from_id, to_id, edge_type }

    Unresolved calls (stdlib, external libraries) are skipped — we only
    graph internal dependencies.
    """
    edges = []
    seen  = set()

    for entity in entities:
        from_id = entity["entity_id"]

        for raw_call in entity.get("calls", []):
            # Match on the call name, trying the full call then its last segment
            candidates = name_index.get(raw_call)
            if not candidates:
                candidates = name_index.get(raw_call.split(".")[-1])
            if not candidates:
                continue  # external / stdlib call — skip

            for to_entity in candidates:
                to_id = to_entity["entity_id"]
                if from_id == to_id:
                    continue  # skip self-loops
                key = (from_id, to_id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({
                    "from_id":   from_id,
                    "to_id":     to_id,
                    "edge_type": "calls",
                })

    return edges


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------

def _enrich_entity(entity: dict, structured_llm, db=None, repo_id: str = "") -> dict:
    """
    Ask Claude to summarise a single entity.
    structured_llm is a ChatBedrock bound with .with_structured_output(EntitySummary, include_raw=True).
    Returns the entity dict with three new fields added.
    """
    prompt = (
        f"You are analysing a legacy codebase for modernisation.\n\n"
        f"Entity type : {entity.get('type', 'unknown')}\n"
        f"Entity ID   : {entity['entity_id']}\n"
        f"File        : {entity.get('file_path', '')}\n"
        f"Lines       : {entity.get('line_start', '?')}–{entity.get('line_end', '?')}\n"
        f"Calls       : {', '.join(entity.get('calls', [])) or 'none'}\n"
        f"Has tests   : {entity.get('has_tests', False)}\n\n"
        f"Source:\n```\n{entity.get('raw_source', '')[:1500]}\n```\n\n"  # cap at 1500 chars
        f"Provide a structured summary following the schema exactly."
    )

    try:
        from agents.observability import log_usage, extract_bedrock_usage, Timer
        with Timer() as t:
            out = structured_llm.invoke(prompt)
        result = out["parsed"] if isinstance(out, dict) else out
        if db is not None and isinstance(out, dict):
            ti, to = extract_bedrock_usage(out.get("raw"))
            log_usage(db, repo_id, "ingestion", "enrichment", ti, to, t.elapsed, "bedrock")
        return {
            **entity,
            "summary_purpose":           result.purpose,
            "summary_dependencies":      result.dependencies_note,
            "summary_modernisation":     result.modernisation_note,
            "summary_ready":             True,
        }
    except Exception as exc:
        print(f"[graph_builder_agent] LLM enrichment failed for {entity['entity_id']}: {exc}")
        return {
            **entity,
            "summary_purpose":       "",
            "summary_dependencies":  "",
            "summary_modernisation": "",
            "summary_ready":         False,
        }


# ---------------------------------------------------------------------------
# Graph builder agent node
# ---------------------------------------------------------------------------

def graph_builder_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — P1 ingestion pipeline, step 2.

    Reads from state:
        state["entities"]   — list of entity dicts from parser_agent
        state["db"]         — live MongoDB database handle
        state["repo_id"]    — stable repo identifier

    Writes back to state:
        state["edges"]        — list of resolved edge dicts
        state["graph_ready"]  — True once edges are persisted
        + all prior state fields passed through unchanged
    """

    entities  = state["entities"]
    db        = state["db"]
    repo_id   = state["repo_id"]

    print(f"[graph_builder_agent] building graph for {len(entities)} entities...")

    # ------------------------------------------------------------------
    # Step 1 — resolve edges (pure deterministic, no LLM)
    # ------------------------------------------------------------------
    name_index = _build_name_index(entities)
    edges = _resolve_edges(entities, name_index)

    print(f"[graph_builder_agent] resolved {len(edges)} internal edges")

    # ------------------------------------------------------------------
    # Step 2 — LLM enrichment with structured output
    # ------------------------------------------------------------------
    llm            = _get_llm()
    structured_llm = llm.with_structured_output(EntitySummary, include_raw=True)

    enriched_entities = []
    for i, entity in enumerate(entities):
        print(f"[graph_builder_agent] enriching {i+1}/{len(entities)}: {entity['entity_id']}")
        enriched = _enrich_entity(entity, structured_llm, db=db, repo_id=repo_id)
        enriched_entities.append(enriched)

    # ------------------------------------------------------------------
    # Step 3 — persist edges to MongoDB
    # ------------------------------------------------------------------
    if edges:
        # Stamp repo_id on every edge so edges from different repos stay
        # separate (the graph view and QA traversal filter by repo).
        for e in edges:
            e["repo_id"] = repo_id

        edge_col = db[EDGES]
        # Replace only THIS repo's edges, leaving other repos intact.
        edge_col.delete_many({"repo_id": repo_id})
        edge_col.insert_many(edges)
        edge_col.create_index("from_id")
        edge_col.create_index("to_id")
        edge_col.create_index("repo_id")
        # Uniqueness is per-repo: the same (from_id, to_id) pair can exist in
        # different repos, so include repo_id in the unique key.
        edge_col.create_index(
            [("repo_id", 1), ("from_id", 1), ("to_id", 1)], unique=True)

        print(f"[graph_builder_agent] wrote {len(edges)} edges for repo {repo_id}")

    # ------------------------------------------------------------------
    # Step 4 — update entity documents with LLM summaries
    # ------------------------------------------------------------------
    entity_col = db[ENTITIES]
    for entity in enriched_entities:
        entity_col.update_one(
            {"repo_id": repo_id, "entity_id": entity["entity_id"]},
            {"$set": {
                "summary_purpose":       entity.get("summary_purpose", ""),
                "summary_dependencies":  entity.get("summary_dependencies", ""),
                "summary_modernisation": entity.get("summary_modernisation", ""),
                "summary_ready":         entity.get("summary_ready", False),
            }},
        )

    # ------------------------------------------------------------------
    # Step 5 — update metadata
    # ------------------------------------------------------------------
    db[METADATA].update_one(
        {"repo_id": repo_id},
        {"$set": {
            "edge_count":      len(edges),
            "graph_ready":     True,
            "enrichment_done": True,
        }},
    )

    return {
        **state,
        "entities":    enriched_entities,
        "edges":       edges,
        "graph_ready": True,
    }