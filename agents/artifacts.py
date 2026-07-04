"""
Artifact generation — downloadable consulting deliverables as branded PDFs.

Two artifacts, both built on the DETERMINISTIC knowledge-graph data
(risk scores, fan-in/out, dependency structure) with LLM-authored analysis
layered on top in clearly-labelled sections:

  1. Modernization Roadmap   — phase-wise plan ordered by criticality and
                               dependency structure.
  2. Complexity Heatmap      — riskiest / most complex modules, scored from
                               size and dependency density.

The numbers are computed and verifiable; the LLM prose is framed as analysis
so the distinction stays clear in the document itself.
"""

import os
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_LEFT

from db.schema import ENTITIES, EDGES, METADATA

MONGO_GREEN  = colors.HexColor("#00684A")   # forest (readable on white)
MONGO_SLATE  = colors.HexColor("#001E2B")
PURPLE   = colors.HexColor("#A100FF")
LIGHT_GREY   = colors.HexColor("#F2F5F6")
BAND_FILL = {
    "high":   colors.HexColor("#F8D7DA"),
    "medium": colors.HexColor("#FFF3CD"),
    "low":    colors.HexColor("#D4EDDA"),
}

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Data loading + deterministic scoring
# ---------------------------------------------------------------------------

def _entity_name(eid: str) -> str:
    parts = eid.split("::")
    return parts[1] if len(parts) >= 2 else eid


def _load_entities(db, repo_id: str) -> list[dict]:
    return list(db[ENTITIES].find(
        {"repo_id": repo_id},
        {"_id": 0, "entity_id": 1, "type": 1, "file_path": 1,
         "line_start": 1, "line_end": 1, "fan_in": 1, "fan_out": 1,
         "has_tests": 1, "risk_score": 1, "risk_band": 1,
         "summary_purpose": 1, "summary_modernisation": 1},
    ))


def _complexity_score(e: dict) -> int:
    """
    A deterministic complexity proxy (0–100) from data we have:
      - size (lines)        — bigger = more complex
      - fan_out (calls made) — more outgoing calls = more branching/coupling
      - fan_in (callers)     — more dependents = more integration complexity
    This is a structural proxy, not true cyclomatic complexity, and the PDF
    says so. Higher = more complex / harder to modernize safely.
    """
    lines  = max(0, (e.get("line_end", 0) or 0) - (e.get("line_start", 0) or 0))
    fan_out = e.get("fan_out", 0) or 0
    fan_in  = e.get("fan_in", 0) or 0

    # Normalise each into a rough 0..1 then weight
    size_n   = min(lines / 80.0, 1.0)     # ~80 lines saturates
    fanout_n = min(fan_out / 10.0, 1.0)   # ~10 calls saturates
    fanin_n  = min(fan_in / 10.0, 1.0)

    score = 100 * (0.45 * size_n + 0.35 * fanout_n + 0.20 * fanin_n)
    return round(score)


def _phase_plan(entities: list[dict]) -> list[dict]:
    """
    Deterministic phased modernization order: foundational, heavily-depended-on
    entities first (high fan-in), then mid-tier, then leaf nodes. Within the
    overall ordering we group into phases by fan-in tier.
    """
    ranked = sorted(entities, key=lambda e: (-(e.get("fan_in", 0) or 0),
                                             -(e.get("risk_score", 0) or 0)))
    # Tier by fan-in
    phases = {"Phase 1 — Foundational (highest fan-in)": [],
              "Phase 2 — Core logic (moderate fan-in)":  [],
              "Phase 3 — Leaf components (low fan-in)":   []}
    for e in ranked:
        fi = e.get("fan_in", 0) or 0
        if fi >= 4:
            phases["Phase 1 — Foundational (highest fan-in)"].append(e)
        elif fi >= 1:
            phases["Phase 2 — Core logic (moderate fan-in)"].append(e)
        else:
            phases["Phase 3 — Leaf components (low fan-in)"].append(e)
    return [{"phase": k, "entities": v} for k, v in phases.items()]


# ---------------------------------------------------------------------------
# LLM analysis (clearly-framed prose on top of the computed data)
# ---------------------------------------------------------------------------

def _llm_analysis(prompt: str) -> str:
    """Get an analysis paragraph from the LLM. Returns '' on failure."""
    try:
        from langchain_aws import ChatBedrock
        llm = ChatBedrock(
            model_id=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return content.strip()
    except Exception as exc:
        print(f"[artifacts] LLM analysis unavailable: {exc}")
        return ""


# ---------------------------------------------------------------------------
# PDF building blocks
# ---------------------------------------------------------------------------

def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Brand", parent=ss["Title"], textColor=MONGO_SLATE,
                          fontSize=22, spaceAfter=2))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], textColor=PURPLE,
                          fontSize=10, spaceAfter=14))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], textColor=MONGO_GREEN,
                          fontSize=14, spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5,
                          leading=14, alignment=TA_LEFT, spaceAfter=6))
    ss.add(ParagraphStyle("AnalysisLabel", parent=ss["Normal"], fontSize=8,
                          textColor=PURPLE, spaceAfter=2))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], fontSize=8,
                          textColor=colors.grey))
    return ss


def _header(story, ss, title, repo_path):
    story.append(Paragraph("LegacyLens", ss["Brand"]))
    story.append(Paragraph("·  Application Modernization", ss["Sub"]))
    story.append(Paragraph(title, ss["H2"]))
    story.append(Paragraph(
        f"Codebase: {repo_path or 'unknown'}<br/>"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ss["Small"]))
    story.append(Spacer(1, 10))


def _analysis_block(story, ss, text):
    if not text:
        return
    story.append(Paragraph("AI ANALYSIS (interpretation of the computed data above)",
                           ss["AnalysisLabel"]))
    for para in text.split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), ss["Body"]))
    story.append(Spacer(1, 6))


# ---------------------------------------------------------------------------
# Artifact 1 — Modernization Roadmap
# ---------------------------------------------------------------------------

def generate_roadmap_pdf(db, repo_id: str, out_path: str,
                         extra_context: dict | None = None,
                         roadmap_text: str | None = None) -> str:
    entities = _load_entities(db, repo_id)
    meta = db[METADATA].find_one({"repo_id": repo_id}, {"_id": 0}) or {}
    repo_path = meta.get("repo_name") or meta.get("repo_path", "")

    ss = _styles()
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    story = []
    _header(story, ss, "Modernization Roadmap", repo_path)

    story.append(Paragraph(
        "This roadmap sequences modernization by dependency structure: "
        "foundational components that many others depend on are addressed first, "
        "so changes cascade safely. Ordering and risk scores are computed "
        "deterministically from the dependency graph; the analysis prose is "
        "AI-generated interpretation.", ss["Body"]))

    phases = _phase_plan(entities)

    # Summary line
    total = len(entities)
    high = sum(1 for e in entities if e.get("risk_band") == "high")
    untested = sum(1 for e in entities if not e.get("has_tests"))
    story.append(Paragraph(
        f"<b>{total}</b> entities · <b>{high}</b> high-risk · "
        f"<b>{untested}</b> untested", ss["Body"]))
    story.append(Spacer(1, 8))

    for ph in phases:
        if not ph["entities"]:
            continue
        story.append(Paragraph(ph["phase"], ss["H2"]))
        # Table of top entities in this phase (cap to keep readable)
        rows = [["Entity", "File", "Risk", "Fan-in", "Tests"]]
        for e in ph["entities"][:15]:
            rows.append([
                _entity_name(e["entity_id"]),
                (e.get("file_path", "") or "").split("/")[-1],
                f"{e.get('risk_score','?')} ({e.get('risk_band','')})",
                str(e.get("fan_in", 0)),
                "yes" if e.get("has_tests") else "no",
            ])
        t = Table(rows, colWidths=[45*mm, 50*mm, 30*mm, 18*mm, 15*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), MONGO_SLATE),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        if len(ph["entities"]) > 15:
            story.append(Paragraph(
                f"… and {len(ph['entities'])-15} more in this phase.", ss["Small"]))
        story.append(Spacer(1, 6))

    # ---- Stakeholder requirements (if gathered interactively) ----
    if extra_context:
        story.append(Paragraph("Stakeholder Requirements", ss["H2"]))
        for k, v in extra_context.items():
            story.append(Paragraph(
                f"<b>{k.replace('_', ' ').title()}:</b> {v}", ss["Body"]))
        story.append(Spacer(1, 8))

    # ---- Detailed roadmap text (from interactive Q&A) or LLM fallback ----
    if roadmap_text:
        story.append(Paragraph("Detailed Modernization Roadmap", ss["H2"]))
        for para in roadmap_text.split("\n"):
            line = para.strip()
            if not line:
                story.append(Spacer(1, 4))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], ss["H2"]))
            elif line.startswith("**") and line.endswith("**"):
                story.append(Paragraph(f"<b>{line.strip('*')}</b>", ss["Body"]))
            else:
                story.append(Paragraph(line, ss["Body"]))
        story.append(Spacer(1, 6))
    else:
        # Fallback: LLM analysis of the roadmap
        phase_summary = "; ".join(
            f"{ph['phase']}: {len(ph['entities'])} entities" for ph in phases)
        top_names = ", ".join(_entity_name(e["entity_id"])
                              for e in sorted(entities, key=lambda x: -(x.get("fan_in", 0) or 0))[:8])
        analysis = _llm_analysis(
            f"You are a software modernization consultant writing the analysis "
            f"section of a roadmap deliverable. The codebase has {total} entities, "
            f"{high} high-risk, {untested} untested. Phased plan: {phase_summary}. "
            f"The most depended-on components are: {top_names}. "
            f"Write 2 short paragraphs of practical guidance on executing this "
            f"modernization: sequencing rationale, key risks, and what to prioritise. "
            f"Be specific and pragmatic. Plain prose, no headers.")
        _analysis_block(story, ss, analysis)

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Methodology: phase tiers are assigned by fan-in (incoming dependencies) "
        "computed from the dependency graph. Risk scores combine fan-in, missing "
        "tests, size, and fan-out. Figures are deterministic; analysis prose is "
        "AI-generated and should be reviewed.", ss["Small"]))

    doc.build(story)
    return out_path


# ---------------------------------------------------------------------------
# Artifact 2 — Complexity Heatmap
# ---------------------------------------------------------------------------

def generate_complexity_pdf(db, repo_id: str, out_path: str) -> str:
    entities = _load_entities(db, repo_id)
    meta = db[METADATA].find_one({"repo_id": repo_id}, {"_id": 0}) or {}
    repo_path = meta.get("repo_path", "")

    # Compute complexity for each entity
    for e in entities:
        e["complexity"] = _complexity_score(e)
    ranked = sorted(entities, key=lambda e: -e["complexity"])

    ss = _styles()
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    story = []
    _header(story, ss, "Application Complexity Heatmap", repo_path)

    story.append(Paragraph(
        "This heatmap ranks modules by a structural complexity score derived "
        "from size (lines), outgoing calls (branching/coupling), and incoming "
        "dependencies. Higher scores indicate modules that are harder and "
        "riskier to modernize. Scores are computed deterministically; the "
        "analysis prose is AI-generated.", ss["Body"]))
    story.append(Spacer(1, 8))

    # Heatmap-style table of the most complex entities
    rows = [["Module", "File", "Complexity", "Lines", "Fan-out", "Risk band"]]
    for e in ranked[:30]:
        lines = max(0, (e.get("line_end", 0) or 0) - (e.get("line_start", 0) or 0))
        rows.append([
            _entity_name(e["entity_id"]),
            (e.get("file_path", "") or "").split("/")[-1],
            str(e["complexity"]),
            str(lines),
            str(e.get("fan_out", 0)),
            e.get("risk_band", ""),
        ])
    t = Table(rows, colWidths=[40*mm, 45*mm, 24*mm, 16*mm, 18*mm, 22*mm])

    # Build per-row shading by complexity (heatmap effect)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), MONGO_SLATE),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, e in enumerate(ranked[:30], start=1):
        c = e["complexity"]
        # red intensity scales with complexity
        if c >= 70:
            fill = colors.HexColor("#F5B7B1")
        elif c >= 45:
            fill = colors.HexColor("#FAD7A0")
        else:
            fill = colors.HexColor("#D5F5E3")
        style.append(("BACKGROUND", (0, i), (-1, i), fill))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 8))

    # Distribution summary
    hi = sum(1 for e in entities if e["complexity"] >= 70)
    md = sum(1 for e in entities if 45 <= e["complexity"] < 70)
    lo = sum(1 for e in entities if e["complexity"] < 45)
    story.append(Paragraph(
        f"Complexity distribution: <b>{hi}</b> high (≥70), "
        f"<b>{md}</b> moderate (45–69), <b>{lo}</b> low (&lt;45).", ss["Body"]))

    # LLM analysis
    top_complex = ", ".join(
        f"{_entity_name(e['entity_id'])} ({e['complexity']})" for e in ranked[:8])
    analysis = _llm_analysis(
        f"You are a software architect writing the analysis section of a code "
        f"complexity report. The most structurally complex modules are: "
        f"{top_complex}. Distribution: {hi} high, {md} moderate, {lo} low "
        f"complexity. Write 2 short paragraphs on what this complexity profile "
        f"means for modernization: which modules need refactoring attention, "
        f"common risks in high-complexity code, and a pragmatic approach. "
        f"Plain prose, no headers.")
    _analysis_block(story, ss, analysis)

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Methodology: complexity is a structural proxy = 0.45·size + "
        "0.35·fan-out + 0.20·fan-in (normalised). It is not true cyclomatic "
        "complexity, which would require statement-level analysis. Figures are "
        "deterministic; analysis prose is AI-generated and should be reviewed.",
        ss["Small"]))

    doc.build(story)
    return out_path