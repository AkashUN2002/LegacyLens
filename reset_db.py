"""
Reset script — drops all LegacyLens collections so you can re-ingest cleanly.

Run from the project root with your venv active:
    python reset_db.py

Use this whenever a previous ingestion left partial/bad data, or when you
want a completely fresh start.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from db.client import get_db, close_client
from db.schema import ENTITIES, EDGES, METADATA


def reset():
    db = get_db()
    for name in (ENTITIES, EDGES, METADATA, "conversations", "usage"):
        db[name].drop()
        print(f"dropped collection: {name}")

    # Also drop the checkpoint database used by the ingestion pipeline
    try:
        client = db.client
        client.drop_database("legacylens_checkpoints")
        print("dropped checkpoint database: legacylens_checkpoints")
    except Exception as exc:
        print(f"(checkpoint db not dropped: {exc})")

    close_client()
    print("\nReset complete. You can now run a fresh ingestion.")


if __name__ == "__main__":
    reset()