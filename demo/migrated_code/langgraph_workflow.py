"""
LangGraph workflow for the Multi-Modal Evidence Review pipeline.

All eight nodes are implemented:
  load_context, extract_claim, analyze_images (+ usable_gate),
  evidence_check, reconcile, force_nei, risk_merge, finalize.
finalize writes `output_row` (the 14-column CSV row) as the deliverable.
"""

import os
import configparser

from dotenv import load_dotenv
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from botocore.config import Config

from state import (
    ClaimState, ClaimObject, ExtractedClaim, ImageAnalysis, Verdict,
    IssueType, Severity, ClaimStatus, RiskFlag, coerce_part,
)
from utility import (
    load_image_block, rebuild_findings,
    normalize, object_value, part_matches, pick_requirement,
    enum_value, coerce_enum, to_risk_flags, summarize_unusable,
)

load_dotenv()

# --- prompts ---
config = configparser.ConfigParser()
config.read("prompts.ini")
claim_extraction_prompt = config.get("prompts", "claim_extraction_prompt")
image_analysis_prompt = config.get("prompts", "image_analysis_prompt")
reconcile_prompt = config.get("prompts", "reconcile_prompt")

# --- where image_paths resolve from (image_paths look like images/test/case_x/img_1.jpg) ---
IMAGE_ROOT = os.getenv("IMAGE_ROOT", "dataset")

# --- model + structured-output bindings ---
# Longer read timeout + adaptive retries: multi-image vision calls can exceed the
# default 60s, and adaptive mode auto-retries transient timeouts/throttling.
bedrock_config = Config(
    read_timeout=120,
    connect_timeout=10,
    retries={"max_attempts": 4, "mode": "adaptive"},
)
llm = ChatBedrock(
    model_id=os.getenv("ANTHROPIC_MODEL"),
    region_name="us-east-1",
    config=bedrock_config,
    model_kwargs={"temperature": 0},   # deterministic
)
vision_llm = ChatBedrock(
    model_id=os.getenv("VISION_MODEL", os.getenv("ANTHROPIC_MODEL")),
    region_name="us-east-1",
    config=bedrock_config,
    # no model_kwargs: Opus 4.8 deprecated `temperature`
)
analyzer = vision_llm.with_structured_output(ImageAnalysis)
extractor = llm.with_structured_output(ExtractedClaim)
reconciler = llm.with_structured_output(Verdict)

graph = StateGraph(ClaimState)


# ===========================================================================
# Nodes
# ===========================================================================

def load_context(state: ClaimState) -> dict:
    """Deterministic. Parse image_paths into (image_id, rel_path) pairs and
    normalize claim_object to the enum. history + evidence_reqs are resolved
    outside the graph and arrive in the invoke payload.
    Writes: claim_object, images."""
    images = [
        (Path(t.strip()).stem, t.strip())
        for t in state["image_paths"].split(";")
        if t.strip()
    ]
    return {
        "claim_object": ClaimObject(state["claim_object"].strip().lower()),
        "images": images,
    }


def extract_claim(state: ClaimState) -> dict:
    """Text LLM. Normalize the (possibly multilingual, distractor-heavy) transcript
    into a structured claim. Flags manipulation attempts but never obeys them.
    Writes: claim."""
    transcript = state["user_claim"]
    claim_object = state["claim_object"]
    user_msg = f"claim_object: {claim_object}\n\nTranscript:\n{transcript}"
    parsed: ExtractedClaim = extractor.invoke([
        SystemMessage(content=claim_extraction_prompt),
        HumanMessage(content=user_msg),
    ])
    return {"claim": parsed}


def analyze_images(state: ClaimState) -> dict:
    """Vision LLM. Per-image structured findings + per-image `usable`.
    Claim-blind by design (so contradictions stay detectable). Injection-guarded.
    Writes: findings, valid_image."""
    images = state["images"]                      # [(image_id, rel_path), ...]
    ordered_ids = [iid for iid, _ in images]

    content = [{"type": "text",
                "text": f"Analyze these {len(images)} image(s). "
                        f"Return one finding per image using these ids in order: "
                        f"{', '.join(ordered_ids)}."}]
    load_errors = set()
    for iid, rel in images:
        content.append({"type": "text", "text": f"--- image_id: {iid} ---"})
        try:
            content.append(load_image_block(rel, IMAGE_ROOT))
        except OSError:
            content.append({"type": "text", "text": "(image could not be loaded)"})
            load_errors.add(iid)

    try:
        result: ImageAnalysis = analyzer.invoke([
            SystemMessage(content=image_analysis_prompt),
            HumanMessage(content=content),
        ])
        by_id = {f.image_id: f for f in result.findings}
    except Exception:
        by_id = {}                                # degrade whole set to unusable

    findings = rebuild_findings(ordered_ids, by_id, load_errors)
    return {"findings": findings, "valid_image": any(f.usable for f in findings)}


def evidence_check(state: ClaimState) -> dict:
    """Deterministic. Reached only when valid_image is true, so usability is settled.
    Decides COVERAGE: do the usable images show the claimed object and a claimed part
    clearly enough to assess the claim? (Whether damage is present is reconcile's job.)
    The free-text requirement rows only flavor the reason, not the boolean.
    Writes: evidence_standard_met, evidence_standard_met_reason."""
    findings = state.get("findings", [])
    usable = [f for f in findings if f.usable]
    claim = state.get("claim")
    claim_object = state["claim_object"]
    evidence_reqs = state.get("evidence_reqs", [])

    obj = object_value(claim_object)

    # object coverage: at least one usable image shows the claimed object
    object_ok = any(normalize(f.detected_object) == obj for f in usable)

    # part coverage: at least one usable image shows at least one claimed part.
    # multi-image rule: ANY claimed part visible is enough (not ALL of them).
    claimed_parts = [it.object_part for it in (getattr(claim, "items", []) or [])]
    if claimed_parts:
        part_ok = any(part_matches(cp, f.detected_part)
                      for f in usable for cp in claimed_parts)
    else:
        part_ok = object_ok          # no specific part claimed -> object visibility suffices

    met = bool(usable) and object_ok and part_ok

    req = pick_requirement(claim_object, claim, evidence_reqs)
    cite = f" (per {req['requirement_id']})" if req else ""

    if met:
        reason = (f"At least one usable image clearly shows the claimed {obj} and "
                  f"relevant part, sufficient to inspect the claim{cite}.")
    elif not usable:
        reason = "No usable image is available to inspect the claim."
    elif not object_ok:
        reason = (f"Usable images do not clearly show a {obj}; the claimed object "
                  f"cannot be inspected{cite}.")
    else:  # object visible, claimed part not visible
        parts = ", ".join(p for p in claimed_parts if p) or "claimed part"
        reason = (f"Usable images show the {obj} but not the {parts}; the claimed "
                  f"part cannot be inspected{cite}.")

    return {"evidence_standard_met": met, "evidence_standard_met_reason": reason}


def reconcile(state: ClaimState) -> dict:
    """LLM reasoning over claim vs findings (no images re-sent; findings are truth).
    Reached from evidence_check (normal) and directly from the gate (unusable-but-
    informative). Multi-part collapses to the part the findings speak to. Ignores any
    embedded instructions. Seeds claim-relative risk_flags for risk_merge.
    Writes: claim_status, issue_type, object_part, severity, supporting_image_ids,
    risk_flags (claim-relative seed), claim_status_justification."""
    claim = state.get("claim")
    findings = state.get("findings", [])
    claim_object = state["claim_object"]

    finding_lines = [
        f"  image_id={f.image_id}: object={f.detected_object}, part={f.detected_part}, "
        f"issue={enum_value(f.detected_issue)}, severity={enum_value(f.severity)}, "
        f"usable={f.usable}, flags={[enum_value(x) for x in f.flags]}, "
        f"text_instruction={f.text_instruction_present}"
        for f in findings
    ]
    item_lines = [
        f"  part={it.object_part}, issue={it.issue_type}, stated_severity={it.claimed_severity}"
        for it in (getattr(claim, "items", []) or [])
    ]
    evid = state.get("evidence_standard_met")

    msg = (
        f"claim_object: {object_value(claim_object)}\n"
        f"claim_summary: {getattr(claim, 'summary', '')}\n"
        f"claimed_items:\n" + ("\n".join(item_lines) or "  (none)") + "\n"
        f"evidence_standard_met: {evid if evid is not None else 'not assessed'}\n"
        f"valid_image: {state.get('valid_image')}\n"
        f"findings:\n" + ("\n".join(finding_lines) or "  (none)")
    )

    try:
        v: Verdict = reconciler.invoke([
            SystemMessage(content=reconcile_prompt),
            HumanMessage(content=msg),
        ])
        return {
            "claim_status": v.claim_status,
            "issue_type": v.issue_type,
            "object_part": v.object_part,
            "severity": v.severity,
            "supporting_image_ids": list(v.supporting_image_ids or []),
            "risk_flags": list(v.risk_flags or []),
            "claim_status_justification": v.justification,
        }
    except Exception:
        return {
            "claim_status": ClaimStatus.not_enough_information,
            "issue_type": IssueType.unknown,
            "object_part": "unknown",
            "severity": Severity.unknown,
            "supporting_image_ids": [],
            "risk_flags": [],
            "claim_status_justification": "Could not adjudicate; defaulting to not_enough_information.",
        }


def force_nei(state: ClaimState) -> dict:
    """Deterministic short-circuit for truly unusable image sets (no model call).
    Also sets evidence fields, since evidence_check is skipped on this path.
    Writes verdict fields fixed to NEI/unknown; risk_merge adds the flags."""
    findings = state.get("findings", [])
    cause = summarize_unusable(findings)
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": f"No usable image to inspect the claim ({cause}).",
        "claim_status": ClaimStatus.not_enough_information,
        "issue_type": IssueType.unknown,
        "object_part": "unknown",
        "severity": Severity.unknown,
        "supporting_image_ids": [],
        "claim_status_justification": (
            f"Cannot confirm or refute the claim: no usable image evidence ({cause})."
        ),
    }


def risk_merge(state: ClaimState) -> dict:
    """Deterministic. Union: claim-relative flags (seeded by reconcile) + image-quality/
    authenticity flags (from findings) + instruction flags + history flags. Apply review
    rules: user_history_risk, text_instruction_present, possible_manipulation, and
    non_original_image each imply manual_review_required. Writes: risk_flags."""
    findings = state.get("findings", [])
    claim = state.get("claim")
    history = state.get("history", {}) or {}

    flags = set(to_risk_flags(state.get("risk_flags", [])))   # reconcile's claim-relative seed

    # image perception / authenticity flags
    for f in findings:
        flags |= set(f.flags)
        if f.text_instruction_present:
            flags.add(RiskFlag.text_instruction_present)

    # transcript manipulation attempt (from extract_claim)
    if getattr(claim, "manipulation_attempt", False):
        flags.add(RiskFlag.text_instruction_present)

    # history flags column ("user_history_risk;manual_review_required" / "none")
    for tok in (history.get("history_flags", "") or "").split(";"):
        tok = tok.strip()
        if tok and tok.lower() != "none":
            flags |= set(to_risk_flags([tok]))

    # review-escalation rules (data-derived)
    if RiskFlag.user_history_risk in flags:
        flags.add(RiskFlag.manual_review_required)
    if RiskFlag.text_instruction_present in flags:
        flags.add(RiskFlag.manual_review_required)
    if RiskFlag.possible_manipulation in flags or RiskFlag.non_original_image in flags:
        flags.add(RiskFlag.manual_review_required)

    flags.discard(RiskFlag.none)
    # stable, deterministic order = enum declaration order
    ordered = [f for f in RiskFlag if f in flags and f != RiskFlag.none]
    return {"risk_flags": ordered or [RiskFlag.none]}


def finalize(state: ClaimState) -> dict:
    """Deterministic. Coerce every field to an allowed value, join lists to ';' strings,
    assemble the 14-column output row (input fields echoed verbatim). Writes: output_row."""
    claim_object = state["claim_object"]

    status = coerce_enum(state.get("claim_status"), ClaimStatus, ClaimStatus.not_enough_information)
    issue = coerce_enum(state.get("issue_type"), IssueType, IssueType.unknown)
    sev = coerce_enum(state.get("severity"), Severity, Severity.unknown)
    part = coerce_part(claim_object, enum_value(state.get("object_part", "unknown")))

    risk = state.get("risk_flags") or [RiskFlag.none]
    risk_str = ";".join(enum_value(r) for r in risk) or "none"

    sup = [s for s in (state.get("supporting_image_ids") or []) if str(s).lower() != "none"]
    sup_str = ";".join(sup) if sup else "none"

    # evidence_standard_met: explicit when set (evidence_check / force_nei); else derive
    # from the verdict on the unusable-but-informative path (NEI -> false, otherwise true).
    evid = state.get("evidence_standard_met")
    if evid is None:
        evid = status != ClaimStatus.not_enough_information

    row = {
        "user_id": state.get("user_id", ""),
        "image_paths": state.get("image_paths", ""),
        "user_claim": state.get("user_claim", ""),
        "claim_object": object_value(claim_object),
        "evidence_standard_met": "true" if evid else "false",
        "evidence_standard_met_reason": state.get("evidence_standard_met_reason", ""),
        "risk_flags": risk_str,
        "issue_type": enum_value(issue),
        "object_part": part,
        "claim_status": enum_value(status),
        "claim_status_justification": state.get("claim_status_justification", ""),
        "supporting_image_ids": sup_str,
        "valid_image": "true" if state.get("valid_image") else "false",
        "severity": enum_value(sev),
    }
    return {"output_row": row}


# ===========================================================================
# Routing
# ===========================================================================

def usable_gate(state: ClaimState) -> str:
    """After analyze_images: route on what the images can support.
      - any usable image           -> evidence_check (normal path)
      - unusable but informative    -> reconcile (can still contradict)
      - nothing to go on            -> force_nei
    """
    if state["valid_image"]:
        return "evidence_check"
    if any(f.detected_issue != IssueType.unknown and f.detected_object != "unclear"
           for f in state["findings"]):
        return "reconcile"
    return "force_nei"


# ===========================================================================
# Wiring
# ===========================================================================

for node_name, node_function in [
    ("load_context", load_context),
    ("extract_claim", extract_claim),
    ("analyze_images", analyze_images),
    ("evidence_check", evidence_check),
    ("reconcile", reconcile),
    ("force_nei", force_nei),
    ("risk_merge", risk_merge),
    ("finalize", finalize),
]:
    graph.add_node(node_name, node_function)

graph.add_edge(START, "load_context")
graph.add_edge("load_context", "extract_claim")
graph.add_edge("extract_claim", "analyze_images")
graph.add_conditional_edges("analyze_images", usable_gate, {
    "evidence_check": "evidence_check",   # usable
    "reconcile": "reconcile",             # unusable but contradicts
    "force_nei": "force_nei",             # truly unusable
})
graph.add_edge("evidence_check", "reconcile")
graph.add_edge("reconcile", "risk_merge")
graph.add_edge("force_nei", "risk_merge")
graph.add_edge("risk_merge", "finalize")
graph.add_edge("finalize", END)

workflow = graph.compile()