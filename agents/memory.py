"""
Conversation memory, backed by MongoDB.

Persists chat turns so the QA agent can resolve follow-up questions
("what about its dependencies?") and so the conversation survives a page
reload. Each turn is one document in the `conversations` collection, keyed
by (session_id, repo_id) and ordered by timestamp.

This is the same MongoDB cluster used for entities/edges/vectors — so the
database now also serves as the chatbot's persistent memory.
"""

from datetime import datetime, timezone

CONVERSATIONS = "conversations"

# How many recent turns to feed back into a new question (sliding window).
HISTORY_WINDOW = 5


def ensure_conversation_indexes(db) -> None:
    """Indexes for fast per-session retrieval. Idempotent."""
    col = db[CONVERSATIONS]
    col.create_index([("session_id", 1), ("repo_id", 1), ("ts", 1)])
    col.create_index("session_id")


def append_turn(db, session_id: str, repo_id: str,
                question: str, answer: str,
                intent: str = "", target_entity: str = "",
                cited_entities: list[str] | None = None) -> None:
    """Append one Q&A turn to the conversation."""
    db[CONVERSATIONS].insert_one({
        "session_id":     session_id,
        "repo_id":        repo_id,
        "ts":             datetime.now(timezone.utc),
        "question":       question,
        "answer":         answer,
        "intent":         intent,
        "target_entity":  target_entity or "",
        "cited_entities": cited_entities or [],
    })


def load_history(db, session_id: str, repo_id: str,
                 limit: int = HISTORY_WINDOW) -> list[dict]:
    """
    Return the most recent `limit` turns for this session+repo, oldest first
    (so they read naturally as a transcript).
    """
    cursor = (db[CONVERSATIONS]
              .find({"session_id": session_id, "repo_id": repo_id},
                    {"_id": 0})
              .sort("ts", -1)
              .limit(limit))
    turns = list(cursor)
    turns.reverse()   # oldest first
    return turns


def load_full_history(db, session_id: str, repo_id: str) -> list[dict]:
    """
    Return the ENTIRE conversation for this session+repo, oldest first.
    Used by the UI to restore the chat panel after a page reload.
    """
    cursor = (db[CONVERSATIONS]
              .find({"session_id": session_id, "repo_id": repo_id},
                    {"_id": 0})
              .sort("ts", 1))
    return list(cursor)


def clear_history(db, session_id: str, repo_id: str) -> None:
    """Delete this session+repo's conversation (for a 'clear chat' action)."""
    db[CONVERSATIONS].delete_many({"session_id": session_id, "repo_id": repo_id})


def format_history_for_prompt(turns: list[dict]) -> str:
    """
    Render recent turns as a compact transcript for the LLM, so it can
    resolve references and follow-ups. Answers are truncated to keep the
    prompt lean.
    """
    if not turns:
        return ""
    lines = []
    for t in turns:
        q = t.get("question", "")
        a = (t.get("answer", "") or "")[:400]
        tgt = t.get("target_entity", "")
        tgt_note = f" (about: {tgt})" if tgt else ""
        lines.append(f"User: {q}{tgt_note}\nAssistant: {a}")
    return "\n\n".join(lines)