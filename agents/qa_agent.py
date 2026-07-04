import os
import time
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

import voyageai
from langchain_aws import ChatBedrock

from db.schema import ENTITIES, EDGES
from agents.mcp_context import get_mcp_context
from agents.memory import (
    load_history, append_turn, format_history_for_prompt,
)
from agents.observability import (
    log_usage, extract_bedrock_usage, extract_voyage_usage, Timer,
)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"


def _get_llm() -> ChatBedrock:
    model_id = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)
    region   = os.environ.get("AWS_REGION", "us-east-1")
    return ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={"temperature": 0},
    )


def _get_voyage_client() -> voyageai.Client:
    return voyageai.Client()


EMBED_MODEL    = "voyage-code-3"
VECTOR_INDEX   = "entity_vector_index"   # name of the Atlas Vector Search index
VECTOR_FIELD   = "embedding"
TOP_K          = 6                       # how many entities to retrieve per query


# ---------------------------------------------------------------------------
# Resilient structured-output invocation
# ---------------------------------------------------------------------------

def _is_transient_tooluse_error(exc: Exception) -> bool:
    """
    True for Bedrock/Claude tool-use hiccups that usually succeed on retry,
    e.g. ModelErrorException: "Model produced invalid sequence as part of
    ToolUse", plus throttling / transient service errors.
    """
    msg = str(exc).lower()
    return (
        "invalid sequence" in msg
        or "tooluse" in msg
        or "modelerror" in msg
        or "throttl" in msg
        or "service unavailable" in msg
        or "timed out" in msg
    )


def _invoke_structured(structured_llm, prompt: str, retries: int = 3):
    """
    Invoke a structured-output LLM, retrying transient Bedrock tool-use errors
    with a short backoff. Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return structured_llm.invoke(prompt)
        except Exception as exc:
            last_exc = exc
            if _is_transient_tooluse_error(exc) and attempt < retries - 1:
                print(f"[qa_agent] transient tool-use error "
                      f"(attempt {attempt + 1}/{retries}) — retrying: {exc}")
                time.sleep(0.6 * (attempt + 1))
                continue
            raise
    raise last_exc


def _plain_answer_fallback(prompt: str) -> str:
    """
    Last-resort plain-text answer when structured tool-use output keeps failing.
    Uses a normal (non-tool) LLM call, which is unaffected by the tool-use bug.
    """
    try:
        resp = _get_llm().invoke(
            prompt + "\n\nRespond in clear plain prose (no JSON, no tools)."
        )
        content = getattr(resp, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        return content.strip() or "I couldn't produce a grounded answer for that."
    except Exception as exc:
        return ("I couldn't answer that due to a temporary model error. "
                f"Please try again. ({exc})")


# ---------------------------------------------------------------------------
# Pydantic models — structured outputs
# ---------------------------------------------------------------------------

class QueryIntent(BaseModel):
    """
    Classification of what the user is asking, produced by Claude.
    Drives which retrieval strategy the agent uses.
    """
    intent: Literal["dependency", "risk", "explain", "search", "impact"] = Field(
        description=(
            "The kind of question. "
            "'dependency' = what depends on X / what does X depend on. "
            "'risk' = riskiest / hardest to change modules. "
            "'explain' = what does X do. "
            "'search' = find code matching a description. "
            "'impact' = what breaks if I change/remove X."
        )
    )
    target_entity: Optional[str] = Field(
        default=None,
        description=(
            "If the question names a specific function, class, or module, "
            "the name of that entity. Otherwise null."
        )
    )


class CitedAnswer(BaseModel):
    """
    The final answer Claude produces, with explicit source attribution.
    Every claim should be traceable to one of the cited entities.
    """
    answer: str = Field(
        description="The natural language answer to the user's question."
    )
    cited_entities: list[str] = Field(
        description="entity_ids of the code entities this answer is based on."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high = answer fully grounded in retrieved structural data. "
            "low = answer relies on AI interpretation that should be verified."
        )
    )


# ---------------------------------------------------------------------------
# Step 1 — intent classification
# ---------------------------------------------------------------------------

def _classify_intent(question: str, structured_llm, history_text: str = "",
                     db=None, repo_id: str = "") -> QueryIntent:
    history_block = ""
    if history_text:
        history_block = (
            f"Recent conversation (for resolving follow-ups and pronouns like "
            f"'it', 'that', 'this one'):\n{history_text}\n\n"
            f"If the new question refers back to an entity discussed above "
            f"(e.g. 'what about its dependencies?'), set target_entity to that "
            f"entity's name.\n\n"
        )
    prompt = (
        f"Classify this question about a legacy codebase.\n\n"
        f"{history_block}"
        f"New question: {question}\n\n"
        f"Return the intent and, if a specific entity is named or clearly "
        f"referred to from the conversation, the target_entity."
    )
    with Timer() as t:
        try:
            out = _invoke_structured(structured_llm, prompt)
        except Exception as exc:
            print(f"[qa_agent] intent classification failed after retries "
                  f"({exc}) — defaulting to 'search'")
            return QueryIntent(intent="search", target_entity=None)
    # include_raw=True returns {"parsed": obj, "raw": msg, "parsing_error": ...}
    parsed = out["parsed"] if isinstance(out, dict) else out
    if parsed is None:
        # Structured parsing failed — fall back to a safe default intent.
        parsed = QueryIntent(intent="search", target_entity=None)
    if db is not None and isinstance(out, dict):
        ti, to = extract_bedrock_usage(out.get("raw"))
        log_usage(db, repo_id, "query", "intent", ti, to, t.elapsed, "bedrock")
    return parsed


# ---------------------------------------------------------------------------
# Step 2a — vector search (Atlas Vector Search)
# ---------------------------------------------------------------------------

def _vector_search(question: str, db, voyage, repo_id: str) -> list[dict]:
    """
    Embed the question and run Atlas $vectorSearch to find the
    most semantically similar entities.
    """
    with Timer() as t:
        embed_resp = voyage.embed(
            [question],
            model=EMBED_MODEL,
            input_type="query",         # query-side embedding
        )
    q_vec = embed_resp.embeddings[0]
    log_usage(db, repo_id, "query", "embedding",
              extract_voyage_usage(embed_resp), 0, t.elapsed, "voyage")

    pipeline = [
        {
            "$vectorSearch": {
                "index":         VECTOR_INDEX,
                "path":          VECTOR_FIELD,
                "queryVector":   q_vec,
                "numCandidates": 100,
                "limit":         TOP_K,
                "filter":        {"repo_id": repo_id},
            }
        },
        {
            "$project": {
                "_id": 0,
                "entity_id": 1,
                "type": 1,
                "file_path": 1,
                "line_start": 1,
                "line_end": 1,
                "summary_purpose": 1,
                "summary_modernisation": 1,
                "risk_score": 1,
                "risk_band": 1,
                "fan_in": 1,
                "fan_out": 1,
                "has_tests": 1,
                "raw_source": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = []
    try:
        results = list(db[ENTITIES].aggregate(pipeline))
    except Exception as exc:
        print(f"[qa_agent] vector search failed ({exc}) — falling back to plain query")

    # Fallback: vector index may not be built yet, may lack the repo_id filter
    # field, or embeddings may be absent. Return some entities anyway so the
    # answer isn't empty. Prefer the repo's own entities; if the repo_id filter
    # matches nothing (id mismatch), fall back to any entities.
    if not results:
        proj = {
            "_id": 0, "entity_id": 1, "type": 1, "file_path": 1,
            "line_start": 1, "line_end": 1, "summary_purpose": 1,
            "summary_modernisation": 1, "risk_score": 1, "risk_band": 1,
            "fan_in": 1, "fan_out": 1, "has_tests": 1,
        }
        results = list(db[ENTITIES].find({"repo_id": repo_id}, proj)
                       .sort("risk_score", -1).limit(TOP_K))
        if not results:
            # With multiple repos in the collection, do NOT fall back to other
            # repos' entities — that would return data from the wrong codebase.
            print("[qa_agent] no entities for this repo_id — returning empty context")

    return results


# ---------------------------------------------------------------------------
# Step 2b — graph traversal ($graphLookup)
# ---------------------------------------------------------------------------

def _graph_traverse(target_entity: str, db, repo_id: str, direction: str = "downstream") -> list[dict]:
    """
    Use MongoDB $graphLookup to recursively walk the dependency graph
    starting from target_entity.

    target_entity may be a short name (e.g. "normalize") or a full entity_id.
    Edges store full entity_ids, so we first resolve the name to the matching
    entity_id(s) before traversing.

    direction:
        "downstream" — what target_entity depends on (follow from_id → to_id)
        "upstream"   — what depends on target_entity (follow to_id → from_id)
    """
    # ------------------------------------------------------------------
    # Resolve the (possibly short) target name to full entity_id(s).
    # entity_id format is "file::name::line", so we match on the name segment.
    # ------------------------------------------------------------------
    target_ids = _resolve_target_ids(target_entity, db, repo_id)
    if not target_ids:
        return []

    if direction == "downstream":
        connect_from, connect_to, start_field = "to_id", "from_id", "from_id"
    else:
        connect_from, connect_to, start_field = "from_id", "to_id", "to_id"

    pipeline = [
        {"$match": {start_field: {"$in": target_ids}, "repo_id": repo_id}},
        {
            "$graphLookup": {
                "from":             EDGES,
                "startWith":        f"${connect_from}",
                "connectFromField": connect_from,
                "connectToField":   connect_to,
                "as":               "dependency_chain",
                "maxDepth":         5,
                "depthField":       "depth",
                "restrictSearchWithMatch": {"repo_id": repo_id},
            }
        },
    ]

    results = list(db[EDGES].aggregate(pipeline))
    if not results:
        return []

    # Collect connected entity_ids. The "other end" of each matched edge is the
    # direct dependent/dependency; the chain adds the transitive ones.
    connected = set()
    for doc in results:
        # The directly-matched edge's other end
        connected.add(doc.get("from_id"))
        connected.add(doc.get("to_id"))
        for hop in doc.get("dependency_chain", []):
            connected.add(hop.get("from_id"))
            connected.add(hop.get("to_id"))

    # Don't include the target itself in the list of dependents
    connected -= set(target_ids)
    connected.discard(None)

    if not connected:
        return []

    # Fetch the full entity records for those ids (scoped to this repo)
    return list(db[ENTITIES].find(
        {"repo_id": repo_id, "entity_id": {"$in": list(connected)}},
        {
            "_id": 0, "entity_id": 1, "type": 1, "file_path": 1,
            "line_start": 1, "line_end": 1, "risk_score": 1,
            "risk_band": 1, "fan_in": 1, "has_tests": 1,
            "summary_purpose": 1,
        },
    ))


def _resolve_target_ids(target_entity: str, db, repo_id: str) -> list[str]:
    """
    Resolve a target reference to full entity_id(s), scoped to one repo.

    Accepts either a full entity_id ("utility.py::normalize::128") or a short
    name ("normalize" or "Class.method"). Matches against the name segment of
    stored entity_ids so a bare name finds all entities with that name.
    """
    # If it's already a full id, use it directly
    if "::" in target_entity:
        exact = db[ENTITIES].find_one(
            {"repo_id": repo_id, "entity_id": target_entity}, {"entity_id": 1})
        if exact:
            return [target_entity]

    # Otherwise match on the name segment. entity_id = "file::name::line",
    # so the name is the second :: -delimited part. Use a regex anchored on
    # the "::name::" pattern (and also allow "::name" with no line suffix).
    import re
    safe = re.escape(target_entity)
    pattern = rf"::{safe}(::|$)"
    docs = db[ENTITIES].find(
        {"repo_id": repo_id, "entity_id": {"$regex": pattern}},
        {"entity_id": 1},
    )
    ids = [d["entity_id"] for d in docs]

    # Fallback: also try matching the last dotted segment, so "tokenize"
    # matches an entity stored as "Class.tokenize"
    if not ids:
        pattern2 = rf"::[\w.]*\.{safe}(::|$)"
        docs2 = db[ENTITIES].find(
            {"repo_id": repo_id, "entity_id": {"$regex": pattern2}},
            {"entity_id": 1},
        )
        ids = [d["entity_id"] for d in docs2]

    return ids


def _entities_by_name(target_entity: str, db, repo_id: str) -> list[dict]:
    """
    Fetch full entity records for a target name (used to include the target
    itself in dependency answers for context). Scoped to one repo.
    """
    ids = _resolve_target_ids(target_entity, db, repo_id)
    if not ids:
        return []
    return list(db[ENTITIES].find(
        {"repo_id": repo_id, "entity_id": {"$in": ids}},
        {
            "_id": 0, "entity_id": 1, "type": 1, "file_path": 1,
            "line_start": 1, "line_end": 1, "risk_score": 1,
            "risk_band": 1, "fan_in": 1, "fan_out": 1, "has_tests": 1,
            "summary_purpose": 1,
        },
    ))


# ---------------------------------------------------------------------------
# Step 2c — risk ranking
# ---------------------------------------------------------------------------

def _top_risk_entities(db, repo_id: str, limit: int = 10) -> list[dict]:
    """Return the highest-risk entities, sorted descending."""
    return list(db[ENTITIES].find(
        {"repo_id": repo_id},
        {
            "_id": 0, "entity_id": 1, "file_path": 1, "risk_score": 1,
            "risk_band": 1, "fan_in": 1, "has_tests": 1,
            "summary_modernisation": 1, "risk_factors": 1,
        },
    ).sort("risk_score", -1).limit(limit))


# ---------------------------------------------------------------------------
# Step 3 — answer synthesis
# ---------------------------------------------------------------------------

def _synthesise_answer(question: str, context: list[dict], structured_llm,
                       mcp_context: str = "", history_text: str = "",
                       db=None, repo_id: str = "") -> CitedAnswer:
    """
    Hand the retrieved context to Claude and get a cited answer.
    The context entities ARE the ground truth — Claude must base
    its answer on them and cite the entity_ids it used.

    mcp_context (optional) holds aggregate/structural facts gathered live from
    MongoDB via the MCP server — totals, breakdowns, counts. It complements the
    per-entity retrieval with whole-codebase numbers.
    """
    context_blocks = []
    for e in context:
        rel = e.get("relationship")
        rel_line = f"\n  relationship: {rel}" if rel else ""
        block = (
            f"- entity_id: {e.get('entity_id')}{rel_line}\n"
            f"  file: {e.get('file_path')} "
            f"(lines {e.get('line_start', '?')}–{e.get('line_end', '?')})\n"
            f"  risk_score: {e.get('risk_score', 'n/a')} ({e.get('risk_band', 'n/a')})\n"
            f"  fan_in: {e.get('fan_in', 'n/a')}, has_tests: {e.get('has_tests', 'n/a')}\n"
            f"  purpose: {e.get('summary_purpose', '')}\n"
            f"  modernisation_note: {e.get('summary_modernisation', '')}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks) if context_blocks else "No matching entities found."

    # Optional aggregate context from the MongoDB MCP server
    mcp_block = ""
    if mcp_context:
        mcp_block = (
            f"\n\nAggregate facts queried live from MongoDB (use these for "
            f"whole-codebase totals and breakdowns):\n{mcp_context}\n"
        )

    # Optional conversation history for follow-up continuity
    history_block = ""
    if history_text:
        history_block = (
            f"\n\nRecent conversation (for context and follow-ups):\n"
            f"{history_text}\n"
        )

    prompt = (
        f"You are LegacyLens, an assistant that answers questions about a "
        f"legacy codebase using the structural data provided below.\n\n"
        f"Rules:\n"
        f"- Base per-entity claims on the retrieved entities. Do not invent dependencies.\n"
        f"- For totals, counts, and breakdowns, use the aggregate facts section.\n"
        f"- Use the conversation history to understand follow-up questions and "
        f"references like 'it' or 'that', but base factual claims on the data.\n"
        f"- Always cite the entity_ids you used in cited_entities.\n"
        f"- Numbers like risk_score and fan_in are verified facts — use them directly.\n"
        f"- Set confidence to 'high' only if the answer is well grounded in the data.\n"
        f"{history_block}\n"
        f"Question: {question}\n\n"
        f"Retrieved entities:\n{context_text}"
        f"{mcp_block}\n"
        f"Answer the question following the schema."
    )

    with Timer() as t:
        try:
            out = _invoke_structured(structured_llm, prompt)
            parsed = out["parsed"] if isinstance(out, dict) else out
        except Exception as exc:
            print(f"[qa_agent] structured synthesis failed after retries "
                  f"({exc}) — falling back to a plain-text answer")
            out, parsed = None, None

    if parsed is None:
        # Fallback: plain (non-structured) answer so chat still works even when
        # Bedrock tool-use is failing. Cite whatever context we retrieved.
        answer_text = _plain_answer_fallback(prompt)
        return CitedAnswer(
            answer=answer_text,
            cited_entities=[e.get("entity_id") for e in context
                            if e.get("entity_id")][:8],
            confidence="low",
        )

    if db is not None and isinstance(out, dict):
        ti, to = extract_bedrock_usage(out.get("raw"))
        log_usage(db, repo_id, "query", "synthesis", ti, to, t.elapsed, "bedrock")
    return parsed


# ---------------------------------------------------------------------------
# QA agent node
# ---------------------------------------------------------------------------

def qa_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — P2 query pipeline core.

    Reads from state:
        state["question"]   — the user's natural language question
        state["db"]         — live MongoDB database handle
        state["repo_id"]    — stable repo identifier

    Writes back to state:
        state["intent"]          — classified QueryIntent
        state["retrieved"]       — list of entities used as context
        state["answer"]          — natural language cited answer
        state["cited_entities"]  — entity_ids the answer is based on
        state["confidence"]      — high / medium / low
    """

    question = state["question"]
    db       = state["db"]
    repo_id  = state["repo_id"]
    session_id = state.get("session_id", "default")

    llm                 = _get_llm()
    intent_llm          = llm.with_structured_output(QueryIntent, include_raw=True)
    answer_llm          = llm.with_structured_output(CitedAnswer, include_raw=True)
    voyage              = _get_voyage_client()

    # ------------------------------------------------------------------
    # Step 0 — load recent conversation history (persistent memory)
    # ------------------------------------------------------------------
    # Recent turns let the agent resolve follow-ups like "what about its
    # dependencies?" — the history tells it what "its" refers to.
    history = load_history(db, session_id, repo_id)
    history_text = format_history_for_prompt(history)

    # ------------------------------------------------------------------
    # Step 1 — classify intent (history-aware)
    # ------------------------------------------------------------------
    intent = _classify_intent(question, intent_llm, history_text=history_text,
                              db=db, repo_id=repo_id)
    print(f"[qa_agent] intent={intent.intent} target={intent.target_entity}")

    # ------------------------------------------------------------------
    # Step 2 — retrieve context, routing by what the question needs
    # ------------------------------------------------------------------
    # Two deterministic fast paths handle the common, high-value questions
    # exactly and quickly:
    #   - named-entity dependency/impact  -> $graphLookup traversal
    #   - risk ranking                    -> sorted risk query
    # Everything else (whole-codebase questions, listing, ordering, counting,
    # and genuinely unpredictable asks) goes to the MCP agent as PRIMARY
    # retrieval — it composes whatever MongoDB query the question needs, rather
    # than us guessing a fixed strategy. This is what handles "users may ask
    # anything": the open-ended tail is served by a layer that adapts per query.
    retrieved: list[dict] = []
    mcp_context = ""
    mcp_tool_calls: list[dict] = []
    used_deterministic = False

    if intent.intent == "risk":
        retrieved = _top_risk_entities(db, repo_id)
        used_deterministic = True

    elif intent.intent in ("dependency", "impact") and intent.target_entity:
        upstream   = _graph_traverse(intent.target_entity, db, repo_id, direction="upstream")
        downstream = _graph_traverse(intent.target_entity, db, repo_id, direction="downstream")
        for e in upstream:
            e["relationship"] = "depends on target (caller)"
        for e in downstream:
            e["relationship"] = "depended on by target (callee)"
        target_recs = _entities_by_name(intent.target_entity, db, repo_id)
        for e in target_recs:
            e["relationship"] = "the target entity"
        retrieved = target_recs + upstream + downstream
        if upstream or downstream or target_recs:
            used_deterministic = True
        else:
            # Named entity not found in graph — fall through to MCP below
            retrieved = []

    if used_deterministic:
        # Fast path: deterministic retrieval is authoritative. We still attach
        # MCP aggregate context for whole-codebase numbers, but keep it cheap.
        mcp = get_mcp_context(question, repo_id, mode="aggregate")
        mcp_context    = mcp.get("text", "")
        mcp_tool_calls = mcp.get("tool_calls", [])
        if mcp_context:
            print(f"[qa_agent] MCP added {len(mcp_context)} chars "
                  f"({len(mcp_tool_calls)} tool calls) of aggregate context")
    else:
        # Open-ended / whole-codebase / unpredictable question:
        # MCP is the PRIMARY retrieval. It queries MongoDB freely and returns
        # whatever the question needs (lists, orderings, counts, facts).
        print("[qa_agent] routing to MCP as primary retrieval (open-ended question)")
        mcp = get_mcp_context(question, repo_id, mode="retrieve")
        mcp_context    = mcp.get("text", "")
        mcp_tool_calls = mcp.get("tool_calls", [])
        # Also run a light vector search so the answer can cite specific
        # entities with file:line, complementing MCP's free-form result.
        retrieved = _vector_search(question, db, voyage, repo_id)
        if mcp_context:
            print(f"[qa_agent] MCP returned {len(mcp_context)} chars "
                  f"({len(mcp_tool_calls)} tool calls) of primary context")

    # Deduplicate by entity_id
    seen, deduped = set(), []
    for e in retrieved:
        eid = e.get("entity_id")
        if eid and eid not in seen:
            seen.add(eid)
            deduped.append(e)
    retrieved = deduped

    print(f"[qa_agent] retrieved {len(retrieved)} entities as context")

    # ------------------------------------------------------------------
    # Step 3 — synthesise cited answer (history-aware)
    # ------------------------------------------------------------------
    result = _synthesise_answer(question, retrieved, answer_llm,
                                mcp_context=mcp_context,
                                history_text=history_text,
                                db=db, repo_id=repo_id)

    # ------------------------------------------------------------------
    # Step 4 — persist this turn to MongoDB (conversation memory)
    # ------------------------------------------------------------------
    try:
        append_turn(
            db, session_id, repo_id,
            question=question,
            answer=result.answer,
            intent=intent.intent,
            target_entity=intent.target_entity or "",
            cited_entities=result.cited_entities,
        )
    except Exception as exc:
        print(f"[qa_agent] could not persist conversation turn: {exc}")

    return {
        **state,
        "intent":         intent.intent,
        "target_entity":  intent.target_entity,
        "retrieved":      retrieved,
        "answer":         result.answer,
        "cited_entities": result.cited_entities,
        "confidence":     result.confidence,
        "mcp_context":    mcp_context,
        "mcp_tool_calls": mcp_tool_calls,
        "used_deterministic": used_deterministic,
    }