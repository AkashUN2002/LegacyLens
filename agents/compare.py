"""
Migration coverage comparison.

Compares two already-ingested codebases (a legacy repo and its modernized
counterpart) by diffing their knowledge graphs in MongoDB. Answers the central
modernization question: "what did we miss?"

Three layers:
  1. Deterministic matching   — exact + fuzzy name matching across entity sets.
  2. LLM adjudication         — for legacy entities with no clear match, judge
                                whether each was likely renamed/merged (fine) or
                                genuinely dropped (a real coverage gap).
  3. Dependency + metric diff — for matched entities, compare call
                                relationships; plus overall metric comparison.

Requires BOTH repos to be ingested first. The deterministic layer is exact;
the LLM layer is explicitly framed as producing review candidates, not verdicts.
"""

import os
import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock

from db.schema import ENTITIES, EDGES, METADATA


DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

# Fuzzy-match threshold: name similarity at/above this counts as a match.
FUZZY_THRESHOLD = 0.82

# Embedding-match threshold: cosine similarity at/above this counts as a
# semantic match for entities whose NAMES didn't match. Renamed-but-equivalent
# code has similar embeddings even when names differ completely. Set
# conservatively so genuinely unrelated functions (which still share some
# baseline similarity) are NOT falsely paired — a false match hides a real gap,
# which is worse than reporting a gap for review.
EMBED_THRESHOLD = float(os.environ.get("COMPARE_EMBED_THRESHOLD", "0.82"))


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is empty."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_name(entity_id: str) -> str:
    """Extract the name segment from 'file::name::line'."""
    parts = entity_id.split("::")
    return parts[1] if len(parts) >= 2 else entity_id


def _short_name(name: str) -> str:
    """Last dotted segment, lowercased — 'PaymentService.process' -> 'process'."""
    return name.split(".")[-1].lower()


def _load_repo(db, repo_id: str) -> dict:
    """Load a repo's entities and edges keyed for comparison."""
    entities = list(db[ENTITIES].find(
        {"repo_id": repo_id},
        {"_id": 0, "entity_id": 1, "type": 1, "file_path": 1,
         "fan_in": 1, "fan_out": 1, "has_tests": 1, "risk_band": 1,
         "summary_purpose": 1, "embedding": 1},
    ))
    edges = list(db[EDGES].find(
        {"repo_id": repo_id},
        {"_id": 0, "from_id": 1, "to_id": 1},
    ))
    return {"entities": entities, "edges": edges}


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Layer 1 — deterministic matching
# ---------------------------------------------------------------------------

def _match_entities(legacy: list[dict], modern: list[dict]) -> dict:
    """
    Hybrid matching of legacy entities to modern ones:
      1. exact short-name match  (fast, names preserved)
      2. fuzzy name match        (camelCase vs snake_case, minor edits)
      3. embedding match         (semantic — catches RENAMED functions whose
                                   code means the same thing despite a new name)

    Returns matched pairs (each tagged with how it matched), unmatched-legacy
    (potential gaps), and unmatched-modern (additions).
    """
    # Index modern entities by short name for fast exact lookup
    modern_by_short: dict[str, list[dict]] = {}
    for m in modern:
        modern_by_short.setdefault(_short_name(_entity_name(m["entity_id"])), []).append(m)

    matched = []          # (legacy_entity, modern_entity, how)
    name_unmatched = []   # legacy entities that didn't match by name
    used_modern_ids = set()

    # ---- Passes 1 & 2: name-based matching ----
    for leg in legacy:
        leg_name  = _entity_name(leg["entity_id"])
        leg_short = _short_name(leg_name)

        # 1) exact short-name match
        candidates = modern_by_short.get(leg_short, [])
        exact = next((c for c in candidates if c["entity_id"] not in used_modern_ids), None)
        if exact:
            matched.append((leg, exact, "exact"))
            used_modern_ids.add(exact["entity_id"])
            continue

        # 2) fuzzy match across all modern names
        best, best_score = None, 0.0
        for m in modern:
            if m["entity_id"] in used_modern_ids:
                continue
            score = _similarity(leg_short, _short_name(_entity_name(m["entity_id"])))
            if score > best_score:
                best, best_score = m, score
        if best and best_score >= FUZZY_THRESHOLD:
            matched.append((leg, best, f"fuzzy:{best_score:.2f}"))
            used_modern_ids.add(best["entity_id"])
        else:
            name_unmatched.append(leg)

    # ---- Pass 3: embedding-based matching for the name-unmatched leftovers ----
    # This is what catches renamed functions: their code embeddings are similar
    # even though their names differ. We greedily pair each remaining legacy
    # entity with the most semantically similar unused modern entity, if it
    # clears the threshold.
    unmatched_legacy = []
    remaining_modern = [m for m in modern if m["entity_id"] not in used_modern_ids]

    for leg in name_unmatched:
        leg_vec = leg.get("embedding") or []
        if not leg_vec:
            unmatched_legacy.append(leg)
            continue

        best, best_sim = None, 0.0
        for m in remaining_modern:
            if m["entity_id"] in used_modern_ids:
                continue
            sim = _cosine(leg_vec, m.get("embedding") or [])
            if sim > best_sim:
                best, best_sim = m, sim

        if best and best_sim >= EMBED_THRESHOLD:
            matched.append((leg, best, f"semantic:{best_sim:.2f}"))
            used_modern_ids.add(best["entity_id"])
        else:
            unmatched_legacy.append(leg)

    unmatched_modern = [m for m in modern if m["entity_id"] not in used_modern_ids]

    return {
        "matched":          matched,
        "unmatched_legacy": unmatched_legacy,   # potential coverage gaps
        "unmatched_modern": unmatched_modern,   # additions
    }


# ---------------------------------------------------------------------------
# Layer 2 — LLM adjudication of unmatched-legacy entities
# ---------------------------------------------------------------------------

class GapVerdict(BaseModel):
    """Per-entity judgment on whether a legacy entity was dropped or relocated."""
    likely_status: str = Field(
        description=("One of: 'dropped' (genuinely missing, a real coverage gap), "
                     "'renamed_or_merged' (likely exists under a different name), "
                     "'trivial' (getter/setter/boilerplate, low concern).")
    )
    reason: str = Field(description="One short sentence explaining the judgment.")


def _get_llm():
    return ChatBedrock(
        model_id=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def _adjudicate_gaps(unmatched_legacy: list[dict], modern: list[dict],
                     max_items: int = 25) -> list[dict]:
    """
    For each unmatched legacy entity, ask the LLM whether it was likely dropped,
    renamed/merged, or trivial — given the list of modern entity names as
    context. Returns enriched gap records. Capped to keep cost/latency bounded.
    """
    if not unmatched_legacy:
        return []

    llm = _get_llm().with_structured_output(GapVerdict)
    modern_names = sorted({_entity_name(m["entity_id"]) for m in modern})
    modern_names_block = ", ".join(modern_names[:300])

    results = []
    for leg in unmatched_legacy[:max_items]:
        name = _entity_name(leg["entity_id"])
        prompt = (
            f"During a codebase modernization, this entity from the LEGACY "
            f"codebase has no exact or close name match in the modernized "
            f"codebase.\n\n"
            f"Legacy entity: {name} ({leg.get('type','')})\n"
            f"Purpose: {leg.get('summary_purpose','(unknown)')}\n"
            f"Fan-in (callers): {leg.get('fan_in', 0)}\n\n"
            f"Modernized codebase entity names (for reference):\n"
            f"{modern_names_block}\n\n"
            f"Judge whether this legacy entity was likely genuinely DROPPED "
            f"(a real coverage gap), likely RENAMED/MERGED into something above, "
            f"or is TRIVIAL boilerplate. Be conservative: only say 'dropped' if "
            f"you don't see a plausible counterpart."
        )
        try:
            verdict = llm.invoke(prompt)
            results.append({
                "entity_id":     leg["entity_id"],
                "name":          name,
                "type":          leg.get("type", ""),
                "fan_in":        leg.get("fan_in", 0),
                "risk_band":     leg.get("risk_band", ""),
                "purpose":       leg.get("summary_purpose", ""),
                "likely_status": verdict.likely_status,
                "reason":        verdict.reason,
            })
        except Exception as exc:
            results.append({
                "entity_id":     leg["entity_id"],
                "name":          name,
                "type":          leg.get("type", ""),
                "fan_in":        leg.get("fan_in", 0),
                "risk_band":     leg.get("risk_band", ""),
                "purpose":       leg.get("summary_purpose", ""),
                "likely_status": "unknown",
                "reason":        f"adjudication failed: {exc}",
            })
    # Any beyond the cap are reported without adjudication
    for leg in unmatched_legacy[max_items:]:
        name = _entity_name(leg["entity_id"])
        results.append({
            "entity_id":     leg["entity_id"],
            "name":          name,
            "type":          leg.get("type", ""),
            "fan_in":        leg.get("fan_in", 0),
            "risk_band":     leg.get("risk_band", ""),
            "purpose":       leg.get("summary_purpose", ""),
            "likely_status": "not_reviewed",
            "reason":        "beyond adjudication cap",
        })
    return results


# ---------------------------------------------------------------------------
# Layer 3 — dependency + metric coverage
# ---------------------------------------------------------------------------

def _dependency_coverage(matched, legacy_edges, modern_edges) -> list[dict]:
    """
    For matched entity pairs, compare outgoing call relationships (by callee
    short-name). Flags matched entities whose modern version appears to have
    fewer dependencies than the legacy one — a possible dropped behavior.
    """
    # Build name-based adjacency for each side
    def out_names(edges, entities_by_id, eid):
        outs = set()
        for e in edges:
            if e["from_id"] == eid:
                outs.add(_short_name(_entity_name(e["to_id"])))
        return outs

    leg_ids = {}   # not strictly needed but kept for clarity
    flags = []
    for leg, mod, how in matched:
        leg_out = set()
        for e in legacy_edges:
            if e["from_id"] == leg["entity_id"]:
                leg_out.add(_short_name(_entity_name(e["to_id"])))
        mod_out = set()
        for e in modern_edges:
            if e["from_id"] == mod["entity_id"]:
                mod_out.add(_short_name(_entity_name(e["to_id"])))

        missing_calls = leg_out - mod_out
        if missing_calls:
            flags.append({
                "legacy_entity": _entity_name(leg["entity_id"]),
                "modern_entity": _entity_name(mod["entity_id"]),
                "match":         how,
                "missing_calls": sorted(missing_calls),
                "legacy_call_count": len(leg_out),
                "modern_call_count": len(mod_out),
            })
    return flags


def _metric_comparison(db, legacy_id, modern_id, legacy, modern) -> dict:
    """Overall metric comparison between the two codebases."""
    def metrics(entities):
        total = len(entities)
        untested = sum(1 for e in entities if not e.get("has_tests"))
        bands = {b: sum(1 for e in entities if e.get("risk_band") == b)
                 for b in ("high", "medium", "low")}
        return {"entities": total, "untested": untested, "bands": bands}

    leg_m = metrics(legacy)
    mod_m = metrics(modern)
    return {"legacy": leg_m, "modern": mod_m}


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------

def compare_codebases(db, legacy_repo_id: str, modern_repo_id: str,
                      run_llm: bool = True) -> dict:
    """
    Compare two ingested codebases. Returns a structured coverage report.
    """
    legacy = _load_repo(db, legacy_repo_id)
    modern = _load_repo(db, modern_repo_id)

    match_result = _match_entities(legacy["entities"], modern["entities"])

    gaps = []
    if run_llm:
        gaps = _adjudicate_gaps(match_result["unmatched_legacy"], modern["entities"])
    else:
        for leg in match_result["unmatched_legacy"]:
            gaps.append({
                "entity_id": leg["entity_id"],
                "name": _entity_name(leg["entity_id"]),
                "type": leg.get("type", ""),
                "fan_in": leg.get("fan_in", 0),
                "risk_band": leg.get("risk_band", ""),
                "purpose": leg.get("summary_purpose", ""),
                "likely_status": "not_reviewed",
                "reason": "LLM adjudication disabled",
            })

    dep_flags = _dependency_coverage(
        match_result["matched"], legacy["edges"], modern["edges"])

    metrics = _metric_comparison(
        db, legacy_repo_id, modern_repo_id,
        legacy["entities"], modern["entities"])

    # Coverage headline: matched / legacy total
    legacy_total = len(legacy["entities"])
    matched_count = len(match_result["matched"])
    coverage_pct = round(100 * matched_count / legacy_total, 1) if legacy_total else 0.0

    # Count genuine-looking gaps (LLM said 'dropped')
    dropped = [g for g in gaps if g["likely_status"] == "dropped"]

    return {
        "coverage_pct":      coverage_pct,
        "legacy_total":      legacy_total,
        "modern_total":      len(modern["entities"]),
        "matched_count":     matched_count,
        "added_count":       len(match_result["unmatched_modern"]),
        "gap_count":         len(gaps),
        "dropped_count":     len(dropped),
        "gaps":              gaps,
        "added":             [_entity_name(m["entity_id"]) for m in match_result["unmatched_modern"]],
        "dependency_flags":  dep_flags,
        "metrics":           metrics,
    }