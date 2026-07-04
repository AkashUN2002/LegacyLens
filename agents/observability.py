"""
Observability — token, cost, and latency tracking, backed by MongoDB.

Every LLM and embedding call logs one record to the `usage` collection:
repo, phase (ingestion | query), operation, token counts, estimated cost,
and latency. An aggregation layer rolls these up for the dashboard.

Same MongoDB cluster as everything else — so the database also serves as the
observability store. One more role for MongoDB in this project.
"""

import time
from datetime import datetime, timezone

USAGE = "usage"


# ---------------------------------------------------------------------------
# Rough cost rates (USD per 1M tokens). Approximate — for relative insight,
# not billing. Adjust to match your actual model/pricing.
# ---------------------------------------------------------------------------
COST_PER_M = {
    # Bedrock Claude (input, output) per 1M tokens — approximate
    "bedrock_input":   3.00,
    "bedrock_output":  15.00,
    # VoyageAI voyage-code-3 — approximate; embeddings are input-only
    "voyage_input":    0.18,
    "voyage_output":   0.0,
}


def ensure_usage_indexes(db) -> None:
    """Indexes for fast per-repo / per-phase rollups. Idempotent."""
    col = db[USAGE]
    col.create_index([("repo_id", 1), ("phase", 1)])
    col.create_index("ts")


def _cost(input_tokens: int, output_tokens: int, provider: str) -> float:
    """Estimate USD cost for a call given provider rate keys."""
    cin  = COST_PER_M.get(f"{provider}_input", 0.0)
    cout = COST_PER_M.get(f"{provider}_output", 0.0)
    return (input_tokens / 1_000_000) * cin + (output_tokens / 1_000_000) * cout


def log_usage(db, repo_id: str, phase: str, operation: str,
              input_tokens: int = 0, output_tokens: int = 0,
              latency_s: float = 0.0, provider: str = "bedrock") -> None:
    """
    Record one operation's usage. Fails silently — observability must never
    break the actual work.

    phase:     "ingestion" | "query"
    operation: "enrichment" | "embedding" | "intent" | "synthesis" | "mcp"
    provider:  "bedrock" | "voyage"
    """
    try:
        db[USAGE].insert_one({
            "repo_id":       repo_id,
            "phase":         phase,
            "operation":     operation,
            "provider":      provider,
            "input_tokens":  int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens":  int((input_tokens or 0) + (output_tokens or 0)),
            "cost_usd":      round(_cost(input_tokens or 0, output_tokens or 0, provider), 6),
            "latency_s":     round(latency_s, 3),
            "ts":            datetime.now(timezone.utc),
        })
    except Exception as exc:
        print(f"[observability] could not log usage: {exc}")


# ---------------------------------------------------------------------------
# Token extraction helpers — pull usage out of provider responses
# ---------------------------------------------------------------------------

def extract_bedrock_usage(message) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a LangChain AIMessage returned
    by ChatBedrock. LangChain exposes usage in usage_metadata (preferred) or
    response_metadata. Returns (0, 0) if not found.
    """
    # Preferred: usage_metadata on the message
    um = getattr(message, "usage_metadata", None)
    if isinstance(um, dict) and um:
        return int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0))

    # Fallback: response_metadata.usage (Bedrock shape)
    rm = getattr(message, "response_metadata", None) or {}
    usage = rm.get("usage") or rm.get("amazon-bedrock-invocationMetrics") or {}
    if usage:
        ti = usage.get("input_tokens") or usage.get("inputTokenCount") or 0
        to = usage.get("output_tokens") or usage.get("outputTokenCount") or 0
        return int(ti), int(to)

    return 0, 0


def extract_voyage_usage(response) -> int:
    """
    Extract total_tokens from a VoyageAI embed response. Returns 0 if absent.
    """
    try:
        return int(getattr(response, "total_tokens", 0) or 0)
    except Exception:
        return 0


class Timer:
    """Context manager to measure wall-clock latency of a block."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        return False


# ---------------------------------------------------------------------------
# Aggregation for the dashboard
# ---------------------------------------------------------------------------

def repo_summary(db, repo_id: str) -> dict:
    """
    Roll up all usage for a repo: totals, split by phase, and a per-operation
    breakdown. Returns a structured dict for the observability view.
    """
    pipeline = [
        {"$match": {"repo_id": repo_id}},
        {"$group": {
            "_id": {"phase": "$phase", "operation": "$operation", "provider": "$provider"},
            "calls":         {"$sum": 1},
            "input_tokens":  {"$sum": "$input_tokens"},
            "output_tokens": {"$sum": "$output_tokens"},
            "total_tokens":  {"$sum": "$total_tokens"},
            "cost_usd":      {"$sum": "$cost_usd"},
            "latency_s":     {"$sum": "$latency_s"},
        }},
    ]
    rows = list(db[USAGE].aggregate(pipeline))

    summary = {
        "by_operation": [],
        "ingestion": {"tokens": 0, "cost": 0.0, "latency": 0.0, "calls": 0},
        "query":     {"tokens": 0, "cost": 0.0, "latency": 0.0, "calls": 0},
        "total":     {"tokens": 0, "cost": 0.0, "latency": 0.0, "calls": 0},
    }
    for r in rows:
        phase = r["_id"]["phase"]
        op    = r["_id"]["operation"]
        prov  = r["_id"]["provider"]
        rec = {
            "phase":         phase,
            "operation":     op,
            "provider":      prov,
            "calls":         r["calls"],
            "input_tokens":  r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "total_tokens":  r["total_tokens"],
            "cost_usd":      round(r["cost_usd"], 6),
            "latency_s":     round(r["latency_s"], 2),
        }
        summary["by_operation"].append(rec)

        bucket = summary.get(phase)
        if bucket is not None:
            bucket["tokens"]  += r["total_tokens"]
            bucket["cost"]    += r["cost_usd"]
            bucket["latency"] += r["latency_s"]
            bucket["calls"]   += r["calls"]

        summary["total"]["tokens"]  += r["total_tokens"]
        summary["total"]["cost"]    += r["cost_usd"]
        summary["total"]["latency"] += r["latency_s"]
        summary["total"]["calls"]   += r["calls"]

    # Round totals
    for k in ("ingestion", "query", "total"):
        summary[k]["cost"]    = round(summary[k]["cost"], 4)
        summary[k]["latency"] = round(summary[k]["latency"], 2)

    # Recent per-query latency (last 10 query-phase calls) for a trend view
    recent = list(db[USAGE]
                  .find({"repo_id": repo_id, "phase": "query"},
                        {"_id": 0, "operation": 1, "total_tokens": 1,
                         "latency_s": 1, "cost_usd": 1, "ts": 1})
                  .sort("ts", -1).limit(20))
    recent.reverse()
    summary["recent_queries"] = recent

    return summary