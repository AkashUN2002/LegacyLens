"""
Inspect graph-readiness per repo.

You may have several repos ingested into the same collections. This script
lists every repo_id present and, for each, checks whether its entities have
the fields the knowledge graph needs (risk_band, fan_in) and whether edges
exist between them.

Run from the project root:
    python diagnose_graph.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from db.client import get_db, close_client
from db.schema import ENTITIES, EDGES, METADATA


def diagnose():
    db = get_db()

    # All repos present (from metadata and from entities themselves)
    meta_repos = {m["repo_id"]: m for m in db[METADATA].find({}, {"_id": 0})}
    entity_repo_ids = db[ENTITIES].distinct("repo_id")

    print("=" * 64)
    print("REPOS PRESENT")
    print("=" * 64)
    print(f"  repo_ids in metadata: {list(meta_repos.keys())}")
    print(f"  repo_ids on entities: {entity_repo_ids}")
    print()

    # Entities with NO repo_id at all (the old bug)
    no_repo = db[ENTITIES].count_documents({"repo_id": {"$exists": False}})
    if no_repo:
        print(f"  ⚠ {no_repo} entities have NO repo_id field "
              f"(ingested before the repo_id fix — re-ingest these)")
        print()

    # Per-repo breakdown
    for rid in entity_repo_ids:
        print("=" * 64)
        print(f"REPO: {rid}")
        meta = meta_repos.get(rid, {})
        if meta:
            print(f"  path: {meta.get('repo_path', '?')}")
        print("=" * 64)

        total = db[ENTITIES].count_documents({"repo_id": rid})
        with_band = db[ENTITIES].count_documents(
            {"repo_id": rid, "risk_band": {"$exists": True}})
        with_fanin = db[ENTITIES].count_documents(
            {"repo_id": rid, "fan_in": {"$exists": True}})

        # Risk band distribution
        bands = {}
        for b in ("high", "medium", "low"):
            bands[b] = db[ENTITIES].count_documents({"repo_id": rid, "risk_band": b})

        # Edges between this repo's entities
        ids = set(db[ENTITIES].distinct("entity_id", {"repo_id": rid}))
        edge_count = 0
        for e in db[EDGES].find({}, {"_id": 0, "from_id": 1, "to_id": 1}):
            if e["from_id"] in ids and e["to_id"] in ids:
                edge_count += 1

        print(f"  entities:            {total}")
        print(f"  with risk_band:      {with_band}")
        print(f"  with fan_in:         {with_fanin}")
        print(f"  band distribution:   high={bands['high']}  "
              f"medium={bands['medium']}  low={bands['low']}")
        print(f"  edges between them:  {edge_count}")
        print()

        # Diagnosis
        if total == 0:
            print("  -> no entities; nothing to graph")
        elif with_band == 0:
            print("  -> entities lack risk_band — ingested before risk scoring. RE-INGEST.")
        elif bands["high"] == 0 and bands["medium"] == 0:
            print("  -> all entities are LOW risk; default graph filter hides them.")
            print("     Toggle 'Show entire graph' or add 'low' to risk bands.")
        elif edge_count == 0:
            print("  -> no edges; graph will show nodes but no connections.")
        else:
            print("  -> looks graph-ready (has risk bands and edges).")
        print()

    close_client()


if __name__ == "__main__":
    diagnose()