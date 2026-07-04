"""
Chat view (Phase 2).

The conversational interface to an ingested codebase. Each user message
runs the query pipeline (P2) and renders the cited answer, the entities
it was grounded in, and a confidence badge.

This is the demo centrepiece — the answers, the file/line citations, and
the confidence labels are what make the "trust but verify" story land.
"""

import streamlit as st

from graph.query_pipeline import run_query
from db.schema import ENTITIES
from agents.memory import load_full_history, clear_history


# Suggested starter questions — shown as clickable chips before the first message.
STARTER_QUESTIONS = [
    "What are the riskiest modules to modernise?",
    "What depends on the payment processor?",
    "Which functions have no test coverage?",
    "Explain what the main service class does.",
]

CONFIDENCE_COLORS = {
    "high":   "🟢",
    "medium": "🟡",
    "low":    "🔴",
}


def render_chat_view(db, repo_id):
    # Stable session id per repo, so reloading the page restores this repo's
    # conversation from MongoDB. (One persistent conversation per codebase.)
    session_id = f"session::{repo_id}"
    st.session_state.session_id = session_id

    # On first render of this repo's chat, restore persisted history from
    # MongoDB so the conversation survives a page reload.
    if not st.session_state.get("history_restored_for") == repo_id:
        restored = load_full_history(db, session_id, repo_id)
        st.session_state.chat_history = [
            {
                "role": "user" if turn_part == "user" else "assistant",
                "content": (turn["question"] if turn_part == "user" else turn["answer"]),
                "meta": None if turn_part == "user" else {
                    "cited_entities": turn.get("cited_entities", []),
                    "confidence": "",      # not re-rendered for restored turns
                    "intent": turn.get("intent", ""),
                    "retrieved": [],
                    "mcp_context": "",
                    "mcp_tool_calls": [],
                    "used_deterministic": False,
                },
            }
            for turn in restored
            for turn_part in ("user", "assistant")
        ]
        st.session_state.history_restored_for = repo_id

    col_h, col_clear = st.columns([4, 1])
    with col_h:
        st.header("Chat with your codebase")
    with col_clear:
        if st.button("Clear chat", use_container_width=True):
            clear_history(db, session_id, repo_id)
            st.session_state.chat_history = []
            st.rerun()

    # ------------------------------------------------------------------
    # Starter chips (only before any conversation)
    # ------------------------------------------------------------------
    if not st.session_state.chat_history:
        st.caption("Try one of these, or ask your own:")
        cols = st.columns(2)
        for i, q in enumerate(STARTER_QUESTIONS):
            if cols[i % 2].button(q, use_container_width=True, key=f"starter_{i}"):
                _submit_question(db, repo_id, q)
                st.rerun()

    # ------------------------------------------------------------------
    # Render existing conversation
    # ------------------------------------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                _render_answer_meta(db, msg["meta"])

    # ------------------------------------------------------------------
    # Input box
    # ------------------------------------------------------------------
    question = st.chat_input("Ask about your codebase…")
    if question:
        _submit_question(db, repo_id, question)
        st.rerun()


# ---------------------------------------------------------------------------
# Question handling
# ---------------------------------------------------------------------------

def _submit_question(db, repo_id, question):
    # Append the user message
    st.session_state.chat_history.append({
        "role":    "user",
        "content": question,
        "meta":    None,
    })

    # Run the query pipeline
    with st.spinner("Searching the codebase…"):
        try:
            session_id = st.session_state.get("session_id", "default")
            result = run_query(question, db, repo_id, session_id=session_id)
        except Exception as exc:
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": f"Something went wrong answering that: {exc}",
                "meta":    None,
            })
            return

    # Append the assistant message with metadata for rich rendering
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": result["answer"],
        "meta": {
            "cited_entities": result["cited_entities"],
            "confidence":     result["confidence"],
            "intent":         result["intent"],
            "retrieved":      result["retrieved"],
            "mcp_context":    result.get("mcp_context", ""),
            "mcp_tool_calls": result.get("mcp_tool_calls", []),
            "used_deterministic": result.get("used_deterministic", False),
        },
    })


# ---------------------------------------------------------------------------
# Rich answer rendering — citations, confidence, sources
# ---------------------------------------------------------------------------

def _render_answer_meta(db, meta):
    confidence = meta.get("confidence", "low")
    badge      = CONFIDENCE_COLORS.get(confidence, "⚪")

    # Confidence + intent line
    st.caption(
        f"{badge} Confidence: {confidence}   ·   "
        f"Query type: {meta.get('intent', 'general')}"
    )

    # ------------------------------------------------------------------
    # MongoDB MCP activity — which tools ran, their args, and what they returned
    # ------------------------------------------------------------------
    tool_calls  = meta.get("mcp_tool_calls", [])
    mcp_text    = meta.get("mcp_context", "")
    deterministic = meta.get("used_deterministic", False)

    # Show how this answer was retrieved
    route_label = (
        "⚡ Deterministic graph/risk query"
        + ("  +  🍃 MongoDB MCP (aggregate context)" if tool_calls else "")
        if deterministic else
        "🍃 MongoDB MCP (primary retrieval)" + ("  +  semantic search" if True else "")
    )
    st.caption(f"Retrieval: {route_label}")

    if tool_calls or mcp_text:
        with st.expander(f"🍃 MongoDB MCP activity ({len(tool_calls)} tool call(s))"):
            if tool_calls:
                for i, tc in enumerate(tool_calls, 1):
                    st.markdown(f"**Tool {i}: `{tc.get('name', 'unknown')}`**")
                    args = tc.get("args", {})
                    if args:
                        st.caption("Arguments")
                        st.code(_pretty(args), language="json")
                    res = tc.get("result", "")
                    if res:
                        st.caption("Result")
                        st.code(res, language="json")
                    st.divider()
            if mcp_text:
                st.caption("Aggregate context passed to the answer")
                st.markdown(mcp_text)

    cited = meta.get("cited_entities", [])
    if not cited:
        return

    # Fetch source locations for the cited entities so every claim is traceable
    docs = list(db[ENTITIES].find(
        {"entity_id": {"$in": cited}},
        {
            "_id": 0, "entity_id": 1, "file_path": 1,
            "line_start": 1, "line_end": 1,
            "risk_score": 1, "risk_band": 1,
        },
    ))

    with st.expander(f"📎 Sources ({len(docs)} entities) — every claim is verifiable"):
        for d in docs:
            band  = d.get("risk_band", "")
            score = d.get("risk_score")
            risk_str = f"  ·  risk {score} ({band})" if score is not None else ""
            st.markdown(
                f"**`{d['entity_id']}`**  \n"
                f"`{d.get('file_path', '?')}` "
                f"lines {d.get('line_start', '?')}–{d.get('line_end', '?')}"
                f"{risk_str}"
            )


def _pretty(obj) -> str:
    """Pretty-print args/results for display."""
    import json
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)