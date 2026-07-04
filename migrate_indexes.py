"""
One-time migration for multi-repo support.

Earlier versions created GLOBAL unique indexes:
    entities: entity_id (unique)
    edges:    from_id + to_id (unique)

Multi-repo support needs PER-REPO uniqueness instead:
    entities: repo_id + entity_id (unique)
    edges:    repo_id + from_id + to_id (unique)

create_index() won't drop a conflicting existing index, so this script removes
the old global ones. The new per-repo indexes are then created by setup_all()
on the next app start (or you can re-ingest).

Run once from the project root:
    python migrate_indexes.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from db.client import get_db, close_client
from db.schema import ENTITIES, EDGES, setup_all


def migrate():
    db = get_db()

    # Drop old global unique indexes if present
    for coll_name, old_index in [(ENTITIES, "entity_id_1"),
                                 (EDGES, "from_id_1_to_id_1")]:
        col = db[coll_name]
        existing = {ix["name"] for ix in col.list_indexes()}
        if old_index in existing:
            col.drop_index(old_index)
            print(f"dropped old index {coll_name}.{old_index}")
        else:
            print(f"(no old index {coll_name}.{old_index} to drop)")

    # Recreate the correct per-repo indexes
    setup_all(db)
    print("\nMigration complete — per-repo indexes are in place.")
    print("Multiple repos can now coexist. Re-ingest any repo whose data was lost.")

    close_client()


if __name__ == "__main__":
    migrate()