import os
from pymongo import MongoClient
from pymongo.database import Database

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_NAME = "legacylens"

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """
    Return a process-wide singleton MongoClient.

    Reuses one connection pool across all agents instead of opening a
    new connection per invocation — important because LangGraph calls
    nodes repeatedly, and each node touches the database.

    Reads the connection string from the MONGO_URI environment variable.
    For the hackathon this is an Atlas M0 free-tier cluster connection string.
    """
    global _client
    if _client is None:
        uri = os.environ.get("MONGO_URI")
        if not uri:
            raise RuntimeError(
                "MONGO_URI environment variable is not set. "
                "Add your Atlas connection string to .env"
            )
        _client = MongoClient(uri)
        # Fail fast if the cluster is unreachable
        _client.admin.command("ping")
        print("[db] connected to MongoDB")
    return _client


def get_db(db_name: str = DEFAULT_DB_NAME) -> Database:
    """
    Return the application database handle.
    This is what gets injected into GraphState["db"].
    """
    return get_client()[db_name]


def close_client() -> None:
    """Close the singleton connection. Call on app shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        print("[db] connection closed")