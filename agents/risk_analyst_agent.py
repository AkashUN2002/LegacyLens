import os
from typing import Any
from datetime import datetime, timezone

import voyageai

from db.schema import ENTITIES, EDGES, METADATA


# ---------------------------------------------------------------------------
# VoyageAI embedding client
# ---------------------------------------------------------------------------

def _get_voyage_client() -> voyageai.Client:
    """
    VoyageAI client. API key picked up from VOYAGE_API_KEY env var.

    max_retries enables the client's built-in wait-and-retry strategy for
    rate-limit (429) errors — important on the free tier, which is capped
    at 3 requests/minute until a payment method is added.
    """
    return voyageai.Client(max_retries=5)


EMBED_MODEL = "voyage-code-3"   # code-specialised embedding model
EMBED_DIM   = 1024              # dimension for voyage-code-3


# ---------------------------------------------------------------------------
# Risk scoring — deterministic, no LLM
# ---------------------------------------------------------------------------

# Weights for each risk factor. Tune these to change scoring behaviour.
WEIGHTS = {
    "fan_in":       0.35,   # how many things depend on this entity
    "fan_out":      0.15,   # how many things this entity depends on
    "no_tests":     0.30,   # untested code is riskier to change
    "size":         0.20,   # large entities are harder to reason about
}


def _compute_fan_in(entity_id: str, edges: list[dict]) -> int:
    """Number of distinct entities that call this one (inbound edges)."""
    return sum(1 for e in edges if e["to_id"] == entity_id)


def _compute_fan_out(entity_id: str, edges: list[dict]) -> int:
    """Number of distinct entities this one calls (outbound edges)."""
    return sum(1 for e in edges if e["from_id"] == entity_id)


def _entity_size(entity: dict) -> int:
    """Lines of code, derived from line_start / line_end."""
    start = entity.get("line_start", 0)
    end   = entity.get("line_end", 0)
    return max(0, end - start)


def _normalise(value: float, max_value: float) -> float:
    """Scale a raw value into 0–1 range. Guards against divide-by-zero."""
    if max_value <= 0:
        return 0.0
    return min(1.0, value / max_value)


def _score_entities(entities: list[dict], edges: list[dict]) -> dict[str, dict]:
    """
    Compute a 0–100 risk score for each entity.

    Returns a dict: entity_id → {
        score, fan_in, fan_out, lines, has_tests, factors
    }

    The score is a weighted, normalised combination of structural factors.
    Higher score = riskier to modernise.
    """
    # First pass — gather raw metrics so we can normalise against the maxima
    raw = {}
    for e in entities:
        eid = e["entity_id"]
        raw[eid] = {
            "fan_in":    _compute_fan_in(eid, edges),
            "fan_out":   _compute_fan_out(eid, edges),
            "lines":     _entity_size(e),
            "has_tests": e.get("has_tests", False),
        }

    # Find maxima for normalisation
    max_fan_in  = max((m["fan_in"]  for m in raw.values()), default=1)
    max_fan_out = max((m["fan_out"] for m in raw.values()), default=1)
    max_lines   = max((m["lines"]   for m in raw.values()), default=1)

    # Second pass — compute weighted score
    scores = {}
    for eid, m in raw.items():
        fan_in_n  = _normalise(m["fan_in"],  max_fan_in)
        fan_out_n = _normalise(m["fan_out"], max_fan_out)
        size_n    = _normalise(m["lines"],   max_lines)
        no_test_n = 0.0 if m["has_tests"] else 1.0

        weighted = (
            WEIGHTS["fan_in"]   * fan_in_n  +
            WEIGHTS["fan_out"]  * fan_out_n +
            WEIGHTS["no_tests"] * no_test_n +
            WEIGHTS["size"]     * size_n
        )

        score = round(weighted * 100)

        scores[eid] = {
            "score":     score,
            "fan_in":    m["fan_in"],
            "fan_out":   m["fan_out"],
            "lines":     m["lines"],
            "has_tests": m["has_tests"],
            "factors": {
                "fan_in_contribution":   round(WEIGHTS["fan_in"]   * fan_in_n  * 100),
                "fan_out_contribution":  round(WEIGHTS["fan_out"]  * fan_out_n * 100),
                "no_tests_contribution": round(WEIGHTS["no_tests"] * no_test_n * 100),
                "size_contribution":     round(WEIGHTS["size"]     * size_n    * 100),
            },
        }

    return scores


def _risk_band(score: int) -> str:
    """Categorise a numeric score into a band for the UI."""
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def _build_embed_text(entity: dict) -> str:
    """
    Compose the text that gets embedded for semantic search.

    Kept deliberately compact. The raw source is truncated hard because
    embedding token cost scales with input length, and the free tier caps
    at 10K tokens/minute. Summaries (when present) carry most of the
    semantic signal anyway, so a short source snippet is enough.
    """
    parts = [
        f"Entity: {entity['entity_id']}",
        f"Type: {entity.get('type', '')}",
        f"Purpose: {entity.get('summary_purpose', '')}",
        f"Dependencies: {entity.get('summary_dependencies', '')}",
        f"Modernisation: {entity.get('summary_modernisation', '')}",
        f"Source: {entity.get('raw_source', '')[:300]}",
    ]
    return "\n".join(p for p in parts if p.strip())


def _embed_entities(entities: list[dict], client: voyageai.Client,
                    db=None, repo_id: str = "") -> dict[str, list[float]]:
    """
    Generate embeddings for all entities, respecting free-tier limits.

    The free tier caps at 3 requests/minute AND 10K tokens/minute. To stay
    under both, we:
      - build small batches bounded by an approximate token budget, and
      - pace requests with a fixed delay between them (default tuned for
        the free tier; set EMBED_DELAY_SECONDS=0 if you have a paid plan).

    Returns dict: entity_id -> embedding vector.
    """
    import os
    import time

    texts = [_build_embed_text(e) for e in entities]
    ids   = [e["entity_id"] for e in entities]

    vectors: dict[str, list[float]] = {}

    # Approximate token budget per request. 1 token ~= 4 chars. We keep each
    # request well under the 10K TPM cap to leave headroom. Configurable.
    token_budget = int(os.environ.get("EMBED_TOKEN_BUDGET", "6000"))
    char_budget  = token_budget * 4

    # Seconds to wait between requests. At 3 RPM the safe spacing is ~21s.
    # Default 21 for the free tier; set to 0 once a payment method is added.
    delay = float(os.environ.get("EMBED_DELAY_SECONDS", "21"))

    # Build token-aware batches
    batches: list[tuple[list[str], list[str]]] = []
    cur_texts, cur_ids, cur_chars = [], [], 0
    for t, eid in zip(texts, ids):
        tlen = len(t)
        # If adding this text would blow the budget, close the current batch
        if cur_texts and cur_chars + tlen > char_budget:
            batches.append((cur_texts, cur_ids))
            cur_texts, cur_ids, cur_chars = [], [], 0
        cur_texts.append(t)
        cur_ids.append(eid)
        cur_chars += tlen
    if cur_texts:
        batches.append((cur_texts, cur_ids))

    total = len(batches)
    print(f"[risk_analyst_agent] embedding in {total} batch(es), "
          f"~{token_budget} token budget each, {delay:.0f}s spacing")

    for bnum, (batch_texts, batch_ids) in enumerate(batches, start=1):
        print(f"[risk_analyst_agent] embedding batch {bnum}/{total} "
              f"({len(batch_texts)} entities)...")

        backoff = 25.0
        for attempt in range(6):
            try:
                from agents.observability import log_usage, extract_voyage_usage, Timer
                with Timer() as _t:
                    resp = client.embed(
                        batch_texts,
                        model=EMBED_MODEL,
                        input_type="document",
                    )
                for eid, vec in zip(batch_ids, resp.embeddings):
                    vectors[eid] = vec
                if db is not None:
                    log_usage(db, repo_id, "ingestion", "embedding",
                              extract_voyage_usage(resp), 0, _t.elapsed, "voyage")
                break
            except Exception as exc:
                msg = str(exc).lower()
                is_rate_limit = ("rate limit" in msg or "429" in msg
                                 or "rpm" in msg or "tpm" in msg)
                if is_rate_limit and attempt < 5:
                    print(f"[risk_analyst_agent] rate limited — waiting {backoff:.0f}s "
                          f"(attempt {attempt + 1}/5)")
                    time.sleep(backoff)
                    backoff *= 1.4
                else:
                    raise

        # Pace before the next request to respect the 3 RPM ceiling
        if delay > 0 and bnum < total:
            print(f"[risk_analyst_agent] pacing — waiting {delay:.0f}s before next batch")
            time.sleep(delay)

    return vectors
    return vectors


# ---------------------------------------------------------------------------
# Risk analyst agent node
# ---------------------------------------------------------------------------

def risk_analyst_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — P1 ingestion pipeline, step 3 (final).

    Reads from state:
        state["entities"]   — enriched entity dicts from graph_builder_agent
        state["edges"]      — resolved edges from graph_builder_agent
        state["db"]         — live MongoDB database handle
        state["repo_id"]    — stable repo identifier

    Writes back to state:
        state["risk_scores"]       — dict entity_id → score record
        state["high_risk_count"]   — number of high-risk entities
        state["embeddings_ready"]  — True once vectors are written
        state["ingestion_done"]    — True, marks P1 complete
    """

    entities = state["entities"]
    edges    = state["edges"]
    db       = state["db"]
    repo_id  = state["repo_id"]

    print(f"[risk_analyst_agent] scoring {len(entities)} entities...")

    # ------------------------------------------------------------------
    # Step 1 — deterministic risk scoring
    # ------------------------------------------------------------------
    scores = _score_entities(entities, edges)

    high_risk_count = sum(1 for s in scores.values() if s["score"] >= 70)
    print(f"[risk_analyst_agent] {high_risk_count} high-risk entities flagged")

    # ------------------------------------------------------------------
    # Step 2 — generate embeddings (non-fatal if it fails)
    # ------------------------------------------------------------------
    # If embedding fails (e.g. a hard rate-limit on the free tier), we do NOT
    # abort ingestion. Risk scoring and the dependency graph are already done
    # and don't need vectors — so risk/dependency questions still work. Only
    # semantic search degrades. Set EMBED_SKIP=1 in .env to skip embeddings
    # entirely on purpose.
    skip_embed = os.environ.get("EMBED_SKIP", "0") == "1"
    vectors: dict[str, list[float]] = {}

    if skip_embed:
        print("[risk_analyst_agent] EMBED_SKIP=1 — skipping embeddings; "
              "semantic search disabled, risk/dependency queries still work")
    else:
        try:
            voyage  = _get_voyage_client()
            vectors = _embed_entities(entities, voyage, db=db, repo_id=repo_id)
        except Exception as exc:
            print(f"[risk_analyst_agent] embeddings failed ({exc}). "
                  f"Continuing without vectors — risk and dependency questions "
                  f"still work; semantic search disabled until re-ingested.")
            vectors = {}

    # ------------------------------------------------------------------
    # Step 3 — write scores + vectors back to each entity document
    # ------------------------------------------------------------------
    entity_col = db[ENTITIES]
    for eid, score_rec in scores.items():
        entity_col.update_one(
            {"repo_id": repo_id, "entity_id": eid},
            {"$set": {
                "risk_score":   score_rec["score"],
                "risk_band":    _risk_band(score_rec["score"]),
                "fan_in":       score_rec["fan_in"],
                "fan_out":      score_rec["fan_out"],
                "risk_factors": score_rec["factors"],
                "embedding":    vectors.get(eid, []),
            }},
        )

    embedded = sum(1 for v in vectors.values() if v)
    print(f"[risk_analyst_agent] wrote scores for {len(scores)} entities "
          f"({embedded} with embeddings)")

    # ------------------------------------------------------------------
    # Step 4 — finalise metadata, mark ingestion complete
    # ------------------------------------------------------------------
    db[METADATA].update_one(
        {"repo_id": repo_id},
        {"$set": {
            "high_risk_count":  high_risk_count,
            "embeddings_ready": True,
            "ingestion_done":   True,
            "ingested_at":      datetime.now(timezone.utc).isoformat(),
        }},
    )

    return {
        **state,
        "risk_scores":      scores,
        "high_risk_count":  high_risk_count,
        "embeddings_ready": True,
        "ingestion_done":   True,
    }