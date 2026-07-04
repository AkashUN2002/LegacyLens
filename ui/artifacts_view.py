"""
Artifacts view — generate and download consulting-deliverable PDFs.

After a codebase is analyzed, the user can produce branded PDF deliverables
built on the deterministic knowledge-graph data:
  · Modernization Roadmap  (interactive Q&A → detailed plan)
  · Application Complexity Heatmap
"""

import os
import json
import tempfile
import streamlit as st

from db.schema import METADATA, ENTITIES, EDGES
from agents.artifacts import (
    generate_roadmap_pdf, generate_complexity_pdf,
    _load_entities, _entity_name, _llm_analysis, _phase_plan,
)


# ---------------------------------------------------------------------------
# Session-state keys for the roadmap Q&A flow
# ---------------------------------------------------------------------------
_RM_STATE     = "roadmap_state"       # "idle" | "gathering" | "generating" | "done"
_RM_QUESTIONS = "roadmap_questions"   # list of {q: str, key: str}
_RM_ANSWERS   = "roadmap_answers"     # dict {key: str}
_RM_ROUND     = "roadmap_round"       # int — which Q&A round
_RM_CONTEXT   = "roadmap_context"     # accumulated context string


def _ensure_roadmap_state():
    """Initialise roadmap session-state keys if not present."""
    for key, default in [
        (_RM_STATE, "idle"),
        (_RM_QUESTIONS, []),
        (_RM_ANSWERS, {}),
        (_RM_ROUND, 0),
        (_RM_CONTEXT, ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


def _build_codebase_summary(db, repo_id: str) -> str:
    """Build a compact text summary of the analysed codebase for the LLM."""
    meta = db[METADATA].find_one({"repo_id": repo_id}, {"_id": 0}) or {}
    entities = _load_entities(db, repo_id)
    high = [e for e in entities if e.get("risk_band") == "high"]
    untested = [e for e in entities if not e.get("has_tests")]
    phases = _phase_plan(entities)

    top_entities = sorted(entities,
                          key=lambda e: -(e.get("fan_in", 0) or 0))[:10]
    top_str = "\n".join(
        f"  - {_entity_name(e['entity_id'])} "
        f"(file: {e.get('file_path','?')}, risk: {e.get('risk_band','?')}, "
        f"fan-in: {e.get('fan_in',0)}, fan-out: {e.get('fan_out',0)}, "
        f"tests: {'yes' if e.get('has_tests') else 'no'})"
        for e in top_entities
    )
    phase_str = "\n".join(
        f"  - {p['phase']}: {len(p['entities'])} entities"
        for p in phases
    )

    return (
        f"Language: {meta.get('language', 'unknown')}\n"
        f"Total entities: {len(entities)}\n"
        f"High-risk entities: {len(high)}\n"
        f"Untested entities: {len(untested)}\n"
        f"Dependency edges: {meta.get('edge_count', '?')}\n"
        f"Phase plan (by fan-in):\n{phase_str}\n"
        f"Most depended-on entities:\n{top_str}"
    )


def _ask_llm_for_questions(codebase_summary: str, previous_answers: str) -> list[dict]:
    """
    Ask the LLM what information it still needs from the human to produce
    a detailed modernization roadmap. Returns a list of question dicts.
    """
    context_block = ""
    if previous_answers:
        context_block = (
            f"\n\nThe human has already provided the following answers from "
            f"previous rounds:\n{previous_answers}\n"
            f"Do NOT re-ask questions that have already been answered."
        )

    prompt = (
        "You are a senior application-modernization consultant. A legacy "
        "codebase has been analysed and here is the analysis summary:\n\n"
        f"{codebase_summary}\n"
        f"{context_block}\n\n"
        "To produce a comprehensive, step-by-step modernization roadmap you "
        "need the MOST CRITICAL information from the human stakeholder. "
        "Focus only on the top decisions that fundamentally shape the roadmap: "
        "target tech stack, target database, deployment target, and "
        "timeline/team constraints.\n\n"
        "Return a JSON array of question objects. Each object has:\n"
        '  {"q": "<the question text>", "key": "<short_snake_case_id>"}\n'
        "Ask EXACTLY 5 or fewer questions. Only ask the most important ones. "
        "Do NOT ask nice-to-have or secondary questions. "
        "If you have everything you need, return an empty array [].\n\n"
        "Return ONLY the JSON array, no other text."
    )

    raw = _llm_analysis(prompt)
    # Parse the JSON array from the LLM response
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(raw[start:end])
            return questions[:5]   # hard cap at 5
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _generate_detailed_roadmap(codebase_summary: str,
                               all_answers: str) -> str:
    """
    Generate the full detailed modernization roadmap using the codebase
    analysis plus all gathered human inputs.
    """
    prompt = (
        "You are a senior application-modernization consultant producing a "
        "comprehensive, client-ready modernization roadmap.\n\n"
        "=== CODEBASE ANALYSIS ===\n"
        f"{codebase_summary}\n\n"
        "=== STAKEHOLDER REQUIREMENTS ===\n"
        f"{all_answers}\n\n"
        "Produce a DETAILED step-by-step modernization roadmap. Structure it "
        "with these sections, using Markdown formatting:\n\n"
        "## Executive Summary\n"
        "A brief overview of the modernization strategy.\n\n"
        "## Current State Assessment\n"
        "Summary of the legacy codebase based on the analysis data.\n\n"
        "## Target Architecture\n"
        "Describe the target state based on the stakeholder requirements — "
        "tech stack, database, deployment, integrations.\n\n"
        "## Modernization Phases\n"
        "For EACH phase, provide:\n"
        "- **Phase name and objective**\n"
        "- **Components/modules to migrate** (reference actual entity names "
        "from the analysis)\n"
        "- **Detailed step-by-step tasks** numbered within the phase\n"
        "- **Dependencies and prerequisites**\n"
        "- **Risk mitigation steps** (especially for high-risk entities)\n"
        "- **Testing strategy** for this phase\n"
        "- **Estimated effort** (relative sizing: S/M/L/XL)\n"
        "- **Definition of Done** for the phase\n\n"
        "## Database Migration Plan\n"
        "Step-by-step plan for data layer migration.\n\n"
        "## Testing & Quality Assurance Strategy\n"
        "Detailed testing approach across all phases.\n\n"
        "## CI/CD Pipeline Setup\n"
        "Steps to establish the build/deploy pipeline for the modernized app.\n\n"
        "## Risk Register & Mitigation\n"
        "Top risks and concrete mitigation actions.\n\n"
        "## Rollback Strategy\n"
        "How to roll back if a phase fails.\n\n"
        "## Success Metrics & KPIs\n"
        "How to measure modernization success.\n\n"
        "Be extremely detailed and specific. Reference actual component names "
        "from the analysis wherever possible. Every phase must have concrete "
        "numbered steps."
    )
    return _llm_analysis(prompt)


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------

def render_artifacts_view(db, repo_id):
    st.header("Deliverables")
    st.write(
        "Generate downloadable consulting artifacts from this codebase's "
        "analysis. Figures (risk, dependencies, complexity, phase ordering) are "
        "computed from the knowledge graph; each report adds AI-written analysis "
        "in clearly-labelled sections."
    )

    meta = db[METADATA].find_one({"repo_id": repo_id}, {"_id": 0}) or {}
    raw_path = meta.get("repo_name") or meta.get("repo_path", "") or "codebase"
    import re
    last = re.split(r"[\\/]", raw_path.rstrip("\\/"))[-1] or "codebase"
    repo_name = re.sub(r"[^A-Za-z0-9._-]", "_", last) or "codebase"

    _ensure_roadmap_state()

    # =========== Top row: two aligned cards ===========
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🗺️ Modernization Roadmap")
        st.caption(
            "Interactive roadmap builder — the AI will ask you questions about "
            "your target architecture, constraints, and goals, then generate a "
            "detailed step-by-step modernization plan."
        )
        # Only show the initial "Build Roadmap" button in the card.
        # Q&A and results go full-width below.
        if st.session_state[_RM_STATE] == "idle":
            if st.button("🚀 Build Roadmap", use_container_width=True,
                         key="start_roadmap"):
                with st.spinner("Analysing codebase and preparing questions…"):
                    summary = _build_codebase_summary(db, repo_id)
                    st.session_state[_RM_CONTEXT] = summary
                    questions = _ask_llm_for_questions(summary, "")
                    st.session_state[_RM_QUESTIONS] = questions
                    st.session_state[_RM_ROUND] = 1
                    st.session_state[_RM_STATE] = "gathering"
                    st.rerun()
        else:
            st.info("Roadmap builder is active — see below.")

    with c2:
        st.subheader("🔥 Complexity Heatmap")
        st.caption("Riskiest, most structurally complex modules, scored from "
                   "size and dependency density.")
        if st.button("Generate heatmap PDF", use_container_width=True, key="gen_heatmap"):
            with st.spinner("Building complexity heatmap…"):
                path = os.path.join(tempfile.gettempdir(),
                                    f"LegacyLens_Complexity_{repo_name}.pdf")
                try:
                    generate_complexity_pdf(db, repo_id, path)
                    with open(path, "rb") as f:
                        st.session_state["heatmap_pdf"] = f.read()
                    st.session_state["heatmap_name"] = os.path.basename(path)
                except Exception as exc:
                    st.error(f"Could not generate heatmap: {exc}")
        if st.session_state.get("heatmap_pdf"):
            st.download_button(
                "⬇ Download heatmap PDF",
                data=st.session_state["heatmap_pdf"],
                file_name=st.session_state.get("heatmap_name", "heatmap.pdf"),
                mime="application/pdf",
                use_container_width=True,
                key="dl_heatmap",
            )

    # =========== Full-width area: Roadmap Q&A (when active) ===========
    if st.session_state[_RM_STATE] != "idle":
        st.divider()
        _render_roadmap_section(db, repo_id, repo_name)

    st.caption(
        "Deliverables are generated on demand. The underlying figures are "
        "deterministic and verifiable against the knowledge graph; AI-written "
        "analysis sections are labelled and intended as a starting point for "
        "expert review."
    )


# ---------------------------------------------------------------------------
# Roadmap interactive Q&A section
# ---------------------------------------------------------------------------

def _render_roadmap_section(db, repo_id, repo_name):
    """Render the roadmap Q&A flow (gathering / generating / done states)."""
    state = st.session_state[_RM_STATE]

    # --- GATHERING: display questions and collect answers ---
    if state == "gathering":
        questions = st.session_state[_RM_QUESTIONS]
        round_num = st.session_state[_RM_ROUND]

        if not questions:
            # LLM has no more questions → proceed to generate
            st.session_state[_RM_STATE] = "generating"
            st.rerun()
            return

        st.info("📋 Please answer the questions below so the AI can build "
                "a detailed roadmap. **Leave blank to skip** — the AI will "
                "use its best judgement for skipped questions.")

        with st.form(key=f"roadmap_form_r{round_num}"):
            answers_this_round = {}
            for q in questions:
                label = q.get("q", "")
                key = q.get("key", label[:30])
                prev = st.session_state[_RM_ANSWERS].get(key, "")
                answers_this_round[key] = st.text_area(
                    label, value=prev, key=f"rm_{key}_{round_num}", height=80,
                    placeholder="Leave blank to skip — AI will decide",
                )

            submitted = st.form_submit_button(
                "Submit & Generate Roadmap", use_container_width=True
            )

        if submitted:
            # Store non-empty answers; mark skipped ones explicitly
            for k, v in answers_this_round.items():
                if v.strip():
                    st.session_state[_RM_ANSWERS][k] = v.strip()
                else:
                    st.session_state[_RM_ANSWERS][k] = "(skipped — use your best judgement)"

            # Go straight to generation — no follow-up rounds
            st.session_state[_RM_STATE] = "generating"
            st.rerun()

    # --- GENERATING: produce the detailed roadmap ---
    elif state == "generating":
        with st.spinner("Generating your detailed modernization roadmap… "
                        "This may take a moment."):
            summary = st.session_state[_RM_CONTEXT]
            all_answers = "\n".join(
                f"- {k}: {v}" for k, v in st.session_state[_RM_ANSWERS].items()
            )
            roadmap_md = _generate_detailed_roadmap(summary, all_answers)
            st.session_state["roadmap_md"] = roadmap_md
            st.session_state[_RM_STATE] = "done"
            st.rerun()

    # --- DONE: display roadmap + download ---
    elif state == "done":
        roadmap_md = st.session_state.get("roadmap_md", "")

        if roadmap_md:
            with st.expander("📖 View full roadmap", expanded=True):
                st.markdown(roadmap_md)

            # Offer download as PDF
            if st.button("Generate roadmap PDF", use_container_width=True,
                         key="gen_roadmap_pdf"):
                with st.spinner("Building roadmap PDF…"):
                    path = os.path.join(tempfile.gettempdir(),
                                        f"LegacyLens_Roadmap_{repo_name}.pdf")
                    try:
                        generate_roadmap_pdf(db, repo_id, path,
                                             extra_context=st.session_state.get(_RM_ANSWERS),
                                             roadmap_text=roadmap_md)
                        with open(path, "rb") as f:
                            st.session_state["roadmap_pdf"] = f.read()
                        st.session_state["roadmap_name"] = os.path.basename(path)
                    except Exception as exc:
                        st.error(f"Could not generate PDF: {exc}")

            if st.session_state.get("roadmap_pdf"):
                st.download_button(
                    "⬇ Download roadmap PDF",
                    data=st.session_state["roadmap_pdf"],
                    file_name=st.session_state.get("roadmap_name", "roadmap.pdf"),
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_roadmap",
                )

        if st.button("🔄 Restart roadmap builder", use_container_width=True,
                     key="restart_roadmap"):
            for k in [_RM_STATE, _RM_QUESTIONS, _RM_ANSWERS, _RM_ROUND,
                      _RM_CONTEXT, "roadmap_md", "roadmap_pdf", "roadmap_name"]:
                st.session_state.pop(k, None)
            _ensure_roadmap_state()
            st.rerun()