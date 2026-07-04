"""
LangGraph checkpointer backed by MongoDB.

A checkpointer persists the graph's state after each node completes.
If the ingestion pipeline crashes partway through a large repo (say,
after the parser but before embeddings finish), it can resume from the
last completed node instead of re-parsing everything from scratch.

Only the ingestion pipeline (P1) uses this. The query pipeline (P2) is
stateless per invocation and needs no checkpointer.
"""

import os

from db.client import get_client

# langgraph-checkpoint-mongodb provides MongoDBSaver
try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
    _HAS_SAVER = True
except ImportError:
    _HAS_SAVER = False


CHECKPOINT_DB_NAME = "legacylens_checkpoints"


def get_checkpointer():
    """
    Return a MongoDBSaver bound to the shared MongoClient, or None if the
    checkpoint library isn't installed (in which case the pipeline still
    runs — it just can't resume after a crash).

    Usage:
        checkpointer = get_checkpointer()
        app = build_ingestion_pipeline(checkpointer=checkpointer)
    """
    if not _HAS_SAVER:
        print(
            "[db] langgraph-checkpoint-mongodb not installed — "
            "ingestion will run without checkpointing"
        )
        return None

    client = get_client()
    saver  = MongoDBSaver(client, db_name=CHECKPOINT_DB_NAME)
    print(f"[db] checkpointer ready (db: {CHECKPOINT_DB_NAME})")
    return saver


def clear_checkpoints(thread_id: str = "ingestion") -> None:
    """
    Remove checkpoints for a given thread. Call this when starting a
    fresh ingestion of a repo you've ingested before, so the pipeline
    doesn't resume from a stale checkpoint.
    """
    client = get_client()
    db = client[CHECKPOINT_DB_NAME]
    for col_name in db.list_collection_names():
        db[col_name].delete_many({"thread_id": thread_id})
    print(f"[db] cleared checkpoints for thread '{thread_id}'")