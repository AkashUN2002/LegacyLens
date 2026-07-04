"""
Pure, model-free helpers for the claim-review pipeline.

  - build_evidence_index / build_history_index : load reference CSVs once, at startup
  - load_image_block / default_finding / rebuild_findings : analyze_images support
"""

import os
import csv
import base64
import mimetypes
from collections import defaultdict

from state import ImageFinding, IssueType, Severity
import io
from PIL import Image

MAX_EDGE = 2048
MAX_BYTES = 4_500_000
BEDROCK_OK = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# ---------------------------------------------------------------------------
# Reference-data indexes (built ONCE before the workflow loop, then passed
# into each graph.invoke payload as `history` and `evidence_reqs`).
# ---------------------------------------------------------------------------

def build_evidence_index(path="dataset/evidence_requirements.csv") -> dict:
    """Group requirements by object; each bucket includes the universal 'all' rows."""
    by_object = defaultdict(list)
    all_rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            obj = row["claim_object"].strip().lower()
            (all_rows if obj == "all" else by_object[obj]).append(row)
    return {obj: rows + all_rows for obj, rows in by_object.items()}


def build_history_index(path="dataset/user_history.csv") -> dict:
    """Map user_id -> its single history row (dict)."""
    with open(path, newline="") as f:
        return {row["user_id"]: row for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# analyze_images helpers
# ---------------------------------------------------------------------------

def _sniff_mime(data: bytes, fallback: str = "image/jpeg") -> str:
    """Detect image type from magic bytes. File extensions lie here: many .jpg
    files in the dataset are actually WebP or PNG, which Bedrock rejects if the
    declared media type does not match the bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return fallback

def load_image_block(rel_path: str, image_root: str) -> dict:
    """Return a Bedrock-safe multimodal image block. Pass through recognized,
    in-spec, under-size images untouched; otherwise normalize via Pillow to JPEG
    (handles unknown formats, oversize files, and corrupt headers)."""
    full = os.path.join(image_root, rel_path)
    with open(full, "rb") as f:
        raw = f.read()

    mime = _sniff_mime(raw, fallback="")

    # fast path: recognized format AND within size -> send original bytes
    if mime in BEDROCK_OK and len(raw) <= MAX_BYTES:
        return {"type": "image", "source_type": "base64",
                "mime_type": mime, "data": base64.b64encode(raw).decode()}

    # everything else: let Pillow decode whatever it is and re-encode as JPEG
    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        s = MAX_EDGE / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    for quality in (85, 70, 55, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= MAX_BYTES:
            break
    return {"type": "image", "source_type": "base64",
            "mime_type": "image/jpeg", "data": base64.b64encode(data).decode()}

def default_finding(image_id: str) -> ImageFinding:
    """Safe 'unusable' finding used when an image fails to load or the model omits it."""
    return ImageFinding(
        image_id=image_id,
        detected_object="unclear",
        detected_part="unknown",
        detected_issue=IssueType.unknown,
        severity=Severity.unknown,
        flags=[],
        text_instruction_present=False,
        usable=False,
    )


def rebuild_findings(ordered_ids: list, by_id: dict, load_errors: set) -> list:
    """Force exactly one finding per input image, correct id, defaults for any gap.

    Guards against the model dropping, reordering, renaming, or inventing findings.
    """
    out = []
    for iid in ordered_ids:
        if iid in load_errors or iid not in by_id:
            out.append(default_finding(iid))
        else:
            f = by_id[iid]
            f.image_id = iid          # correct any id drift from the model
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# evidence_check helpers (pure, no model)
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase and unify separators: 'Front Bumper' / 'front-bumper' -> 'front_bumper'."""
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")


def object_value(claim_object) -> str:
    """Get the plain string for a ClaimObject enum or raw string ('car')."""
    return getattr(claim_object, "value", str(claim_object)).strip().lower()


def part_matches(claimed: str, detected: str) -> bool:
    """Tolerant part match. Returns True when a usable image plausibly shows the claimed
    part well enough to inspect it. Equal, granularity variants ('bumper' vs 'front_bumper'),
    shared head-noun ('left_door' vs 'door'), or a generic whole-object/body shot (which
    shows any part). 'unknown'/empty never match. 'front_bumper' and 'rear_bumper' still
    do NOT match each other (no shared non-generic token)."""
    c, d = normalize(claimed), normalize(detected)
    if not c or not d or c == "unknown" or d == "unknown":
        return False
    if c == d:
        return True
    # a whole-object or body-level shot is treated as covering any specific part
    GENERIC = {"body", "car", "laptop", "package", "box", "exterior", "object"}
    if d in GENERIC:
        return True
    # granularity variant: one string contained in the other
    if c in d or d in c:
        return True
    # shared meaningful token (head noun), but only if positions don't conflict
    POS = {"left", "right", "front", "rear", "back", "upper", "lower", "top", "bottom"}
    STOP = POS | {"side", "the", "a", "of", "outer", "inner"}
    cp, dp = c.split("_"), d.split("_")
    c_pos = {t for t in cp if t in POS}
    d_pos = {t for t in dp if t in POS}
    # if both name a position and they disagree, it's a different part (front vs rear bumper)
    if c_pos and d_pos and not (c_pos & d_pos):
        return False
    ct = {t for t in cp if t and t not in STOP}
    dt = {t for t in dp if t and t not in STOP}
    return bool(ct & dt)


def pick_requirement(claim_object, claim, evidence_reqs):
    """Best-effort: choose the requirement row most relevant to this claim, by token
    overlap between the claim and each row's free-text `applies_to`. Used ONLY to cite
    a requirement_id in the human-readable reason; never affects the boolean.
    Object-specific rows are preferred over universal 'all' rows on ties. Always returns
    a row when any object/all row is present (the universal REQ_* rows are the fallback).
    """
    obj = object_value(claim_object)
    toks = set()
    for it in getattr(claim, "items", []) or []:
        toks |= set(normalize(it.issue_type).split("_"))
        toks |= set(normalize(it.object_part).split("_"))

    best, best_score = None, -1.0
    for req in evidence_reqs or []:
        row_obj = (req.get("claim_object") or "").strip().lower()
        if row_obj not in (obj, "all"):
            continue
        applies = set(normalize(req.get("applies_to", "")).split("_"))
        score = float(len(toks & applies)) + (0.5 if row_obj == obj else 0.0)
        if score > best_score:
            best, best_score = req, score
    return best


# ---------------------------------------------------------------------------
# reconcile / force_nei / risk_merge / finalize helpers (pure, no model)
# ---------------------------------------------------------------------------

from state import RiskFlag, ClaimStatus, IssueType, Severity   # noqa: E402

# Output CSV column order (matches sample_claims.csv exactly)
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def enum_value(x):
    """Return the underlying string for an enum member, or x unchanged."""
    return getattr(x, "value", x)


def coerce_enum(value, enum_cls, default):
    """Snap a value (enum or raw string) to a member of enum_cls, else default."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(enum_value(value))
    except (ValueError, TypeError):
        return default


def to_risk_flags(tokens) -> list:
    """Convert a mix of RiskFlag members / strings into valid RiskFlag members,
    dropping anything that is not a recognized flag."""
    out = []
    for t in tokens or []:
        if isinstance(t, RiskFlag):
            out.append(t)
            continue
        try:
            out.append(RiskFlag(normalize(t)))
        except ValueError:
            pass
    return out


def summarize_unusable(findings) -> str:
    """Human-readable cause string for force_nei reasons."""
    causes = set()
    for f in findings:
        for fl in f.flags:
            causes.add(enum_value(fl))
        if normalize(f.detected_object) in ("other", "unclear"):
            causes.add("no clear inspectable object")
    return ", ".join(sorted(causes)) or "no inspectable image"