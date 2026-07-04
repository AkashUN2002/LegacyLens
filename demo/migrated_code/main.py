"""
Driver: run every claim in an input CSV through the LangGraph workflow and write output.csv.

Usage (from anywhere; paths are anchored to this file):
    python main.py                          # claims.csv -> output.csv
    python main.py dataset/claims.csv out.csv

Each input row becomes one workflow.invoke; the finalize node's `output_row` is collected
and written out in the exact 14-column order. A failure on any single row is caught and
emitted as a safe not_enough_information row so one bad claim never kills the batch.
"""

import os
import csv
import sys

# --- anchor paths to this file so relative cwd never matters ---
BASE = os.path.dirname(os.path.abspath(__file__))          # ...\code
DATASET = os.path.join(os.path.dirname(BASE), "dataset")   # ...\hackerrank-orchestrate-june26\dataset

# The workflow reads IMAGE_ROOT at import time -> must be set BEFORE importing it.
os.environ.setdefault("IMAGE_ROOT", DATASET)

from utility import build_evidence_index, build_history_index, OUTPUT_COLUMNS  # noqa: E402
from langgraph_workflow import workflow                                         # noqa: E402

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATASET, "claims.csv")
OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATASET, "output.csv")


def fallback_row(row: dict, err: str) -> dict:
    """Safe NEI row (echoes inputs) used when a single claim raises."""
    return {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": (row.get("claim_object", "") or "").strip().lower(),
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": f"pipeline error: {err}",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Processing failed; defaulting to not_enough_information.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def run(input_csv: str, output_csv: str) -> None:
    # reference data: built ONCE, passed into each invoke (resolved outside the graph)
    evidence_index = build_evidence_index(os.path.join(DATASET, "evidence_requirements.csv"))
    history_index = build_history_index(os.path.join(DATASET, "user_history.csv"))

    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))   # extra label columns (if any) are ignored

    results = []
    for i, row in enumerate(rows, 1):
        claim_object = (row["claim_object"] or "").strip().lower()
        payload = {
            # the four input fields
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            # pre-resolved context
            "history": history_index.get(row["user_id"], {}),
            "evidence_reqs": evidence_index.get(claim_object, []),
        }
        try:
            final_state = workflow.invoke(payload)
            output_row = final_state["output_row"]
        except Exception as e:
            output_row = fallback_row(row, str(e))

        results.append(output_row)
        print(f"[{i}/{len(rows)}] {row['user_id']} {claim_object:8s} -> "
              f"{output_row['claim_status']} (valid_image={output_row['valid_image']})")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {output_csv}")


if __name__ == "__main__":
    run(INPUT_CSV, OUTPUT_CSV)