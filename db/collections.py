from pymongo.database import Database
from pymongo.operations import SearchIndexModel

# ---------------------------------------------------------------------------
# Collection names — imported by all agents so names live in one place
# ---------------------------------------------------------------------------

ENTITIES = "entities"     # one document per code entity (function/class/method)
EDGES    = "edges"        # one document per dependency edge
METADATA = "metadata"     # one document per ingested repo

# ---------------------------------------------------------------------------
# Vector search configuration
# ---------------------------------------------------------------------------

VECTOR_INDEX = "entity_vector_index"
VECTOR_FIELD = "embedding"
VECTOR_DIM   = 1024          # matches voyage-code-3 output dimension


# ---------------------------------------------------------------------------
# Standard index setup (B-tree indexes)
# ---------------------------------------------------------------------------

def ensure_indexes(db: Database) -> None:
    """
    Create the standard query indexes used across the pipelines.
    Idempotent — safe to call on every startup.

    These power the graph traversal ($graphLookup on from_id/to_id),
    entity lookups, and repo filtering.
    """
    entities = db[ENTITIES]
    entities.create_index("entity_id", unique=True)
    entities.create_index("file_path")
    entities.create_index("calls")
    entities.create_index("repo_id")
    entities.create_index("risk_score")

    edges = db[EDGES]
    edges.create_index("from_id")
    edges.create_index("to_id")
    edges.create_index([("from_id", 1), ("to_id", 1)], unique=True)

    metadata = db[METADATA]
    metadata.create_index("repo_id", unique=True)

    print("[db] standard indexes ensured")


# ---------------------------------------------------------------------------
# Atlas Vector Search index setup
# ---------------------------------------------------------------------------

def ensure_vector_index(db: Database) -> None:
    """
    Create the Atlas Vector Search index on the entities collection.

    This is what powers $vectorSearch in the QA agent. It indexes the
    `embedding` field (a 1024-dim vector from voyage-code-3) and also
    indexes `repo_id` as a filter field so queries can be scoped to one repo.

    Requires an Atlas cluster (vector search is not available on a local
    mongod). On M0 free tier this works but is rate-limited.

    Idempotent — checks whether the index already exists before creating.
    """
    entities = db[ENTITIES]

    # Check existing search indexes
    try:
        existing = {idx["name"] for idx in entities.list_search_indexes()}
    except Exception as exc:
        print(f"[db] could not list search indexes (is this an Atlas cluster?): {exc}")
        return

    if VECTOR_INDEX in existing:
        print(f"[db] vector index '{VECTOR_INDEX}' already exists")
        return

    definition = {
        "fields": [
            {
                "type":          "vector",
                "path":          VECTOR_FIELD,
                "numDimensions": VECTOR_DIM,
                "similarity":    "cosine",
            },
            {
                "type": "filter",
                "path": "repo_id",
            },
        ]
    }

    model = SearchIndexModel(
        definition=definition,
        name=VECTOR_INDEX,
        type="vectorSearch",
    )

    try:
        entities.create_search_index(model=model)
        print(
            f"[db] created vector index '{VECTOR_INDEX}' "
            f"({VECTOR_DIM}-dim, cosine). Note: Atlas builds it asynchronously — "
            f"it may take a minute before queries return results."
        )
    except Exception as exc:
        print(f"[db] failed to create vector index: {exc}")


# ---------------------------------------------------------------------------
# Convenience — full setup
# ---------------------------------------------------------------------------

def setup_all(db: Database) -> None:
    """Run all index setup. Call once after first connecting."""
    ensure_indexes(db)
    ensure_vector_index(db)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_repo_metadata(db: Database, repo_id: str) -> dict | None:
    """Fetch the metadata record for a repo, or None if not yet ingested."""
    return db[METADATA].find_one({"repo_id": repo_id}, {"_id": 0})


def is_repo_ingested(db: Database, repo_id: str) -> bool:
    """
    Check whether a repo has completed ingestion.
    The UI calls this on load to decide whether to show the upload
    screen or jump straight to the chat.
    """
    meta = db[METADATA].find_one({"repo_id": repo_id}, {"ingestion_done": 1})
    return bool(meta and meta.get("ingestion_done"))