"""
Migration coverage view.

Lets the user upload a legacy codebase and its modernized counterpart as .zip
files, analyses each (or reuses it if already analysed), then runs the
graph-vs-graph comparison and presents a coverage report: what appears
preserved, what may have been missed, and where dependencies may have been
dropped.
"""

import zipfile
import streamlit as st

from db.collections import METADATA, is_repo_ingested
from agents.compare import compare_codebases
from agents.parser_agent import _repo_id
from graph.ingestion_pipeline import build_ingestion_pipeline
from db.checkpointer import get_checkpointer
from ui.ingestion_view import _extract_zip


STATUS_BADGE = {
    "dropped":            "🔴 likely dropped",
    "renamed_or_merged":  "🟢 likely renamed/merged",
    "trivial":            "⚪ trivial",
    "not_reviewed":       "⚪ not reviewed",
    "unknown":            "⚪ unknown",
}

# Friendly labels for the ingestion pipeline stages.
_STAGE_LABELS = {
    "parser":        "Parsing source · extracting entities",
    "graph_builder": "Mapping dependencies · enriching with AI",
    "risk_analyst":  "Scoring risk · building search index",
}


def _ensure_ingested(db, uploaded_file, language, role_label):
    """
    Extract an uploaded .zip and make sure it's ingested. If the same codebase
    was already analysed (e.g. on the home screen), reuse it. Returns
    (repo_id, repo_name) or None on failure.
    """
    try:
        repo_path, repo_name = _extract_zip(uploaded_file)
    except (zipfile.BadZipFile, OSError) as exc:
        st.error(f"Could not extract the {role_label} zip: {exc}")
        return None

    repo_id = _repo_id(repo_path)

    # Already analysed — reuse it rather than re-running the pipeline.
    if is_repo_ingested(db, repo_id):
        return repo_id, repo_name

    # Otherwise run the ingestion pipeline now.
    status = st.status(f"Analysing {role_label} codebase…", expanded=True)
    checkpointer = get_checkpointer()
    app = build_ingestion_pipeline(checkpointer=checkpointer)
    initial_state = {"repo_path": repo_path, "language": language, "db": db}
    config = {"configurable": {"thread_id": repo_id}} if checkpointer else {}

    try:
        for chunk in app.stream(initial_state, config=config):
            for node_name in chunk:
                if node_name in _STAGE_LABELS:
                    status.write(f"✓ {_STAGE_LABELS[node_name]}")
        db[METADATA].update_one({"repo_id": repo_id},
                                {"$set": {"repo_name": repo_name}})
        status.update(label=f"{role_label.capitalize()} codebase analysed",
                      state="complete")
    except Exception as exc:
        status.update(label=f"{role_label.capitalize()} analysis failed",
                      state="error")
        st.error(f"Ingestion error ({role_label}): {exc}")
        return None

    return repo_id, repo_name


def render_compare_view(db):
    st.header("Migration coverage")
    st.write(
        "Compare a legacy codebase against its modernized version to find what "
        "may have been missed. Upload both as .zip files — each is analysed "
        "automatically (or reused if already analysed). The graph diff is exact; "
        "the 'likely status' on each gap is an AI judgment meant as a review "
        "aid, not a verdict."
    )

    # Two zip uploaders — legacy and modernized.
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Legacy codebase")
        legacy_zip = st.file_uploader("Legacy (.zip)", type=["zip"],
                                      key="cmp_legacy_zip",
                                      label_visibility="collapsed")
        legacy_lang = st.selectbox("Legacy language", ["python", "java"],
                                   index=1, key="cmp_legacy_lang")
    with c2:
        st.subheader("Modernized codebase")
        modern_zip = st.file_uploader("Modernized (.zip)", type=["zip"],
                                      key="cmp_modern_zip",
                                      label_visibility="collapsed")
        modern_lang = st.selectbox("Modernized language", ["python", "java"],
                                   index=0, key="cmp_modern_lang")

    use_llm = st.toggle("Use AI adjudication (renamed vs dropped)", value=True,
                        help="Off = faster, deterministic only; renamed entities show as gaps.")

    if not st.button("Compare codebases", type="primary"):
        return

    if legacy_zip is None or modern_zip is None:
        st.warning("Upload both a legacy and a modernized .zip to compare.")
        return

    # Analyse (or reuse) each codebase before diffing.
    legacy = _ensure_ingested(db, legacy_zip, legacy_lang, "legacy")
    if legacy is None:
        return
    modern = _ensure_ingested(db, modern_zip, modern_lang, "modernized")
    if modern is None:
        return

    legacy_id, _ = legacy
    modern_id, _ = modern

    if legacy_id == modern_id:
        st.info("Both uploads resolve to the same codebase. Upload two different "
                "codebases to compare.")
        return

    with st.spinner("Diffing the two knowledge graphs…"):
        report = compare_codebases(db, legacy_id, modern_id, run_llm=use_llm)

    # ------------------------------------------------------------------
    # Headline metrics
    # ------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entity coverage", f"{report['coverage_pct']}%",
              help="Legacy entities with a match in the modernized codebase.")
    m2.metric("Matched", f"{report['matched_count']}/{report['legacy_total']}")
    m3.metric("Likely dropped", report["dropped_count"])
    m4.metric("New in modernized", report["added_count"])

    st.divider()

    # ------------------------------------------------------------------
    # Potential gaps (the headline output)
    # ------------------------------------------------------------------
    st.subheader("Potential coverage gaps")
    gaps = report["gaps"]
    if not gaps:
        st.success("Every legacy entity has a match in the modernized codebase.")
    else:
        # Sort: dropped first (most important), then by fan-in (impact)
        order = {"dropped": 0, "unknown": 1, "not_reviewed": 2,
                 "renamed_or_merged": 3, "trivial": 4}
        gaps_sorted = sorted(
            gaps, key=lambda g: (order.get(g["likely_status"], 9), -g.get("fan_in", 0)))
        st.caption("Legacy entities with no clear counterpart, with an AI judgment "
                   "on whether each looks genuinely dropped. Sorted by concern.")
        for g in gaps_sorted:
            badge = STATUS_BADGE.get(g["likely_status"], g["likely_status"])
            with st.expander(f"{badge}  ·  `{g['name']}`  ·  fan-in {g.get('fan_in', 0)}"):
                st.markdown(f"**Type:** {g.get('type','')}")
                if g.get("purpose"):
                    st.markdown(f"**Purpose:** {g['purpose']}")
                st.markdown(f"**Judgment:** {g.get('reason','')}")
                st.caption(f"entity_id: {g['entity_id']}")

    st.divider()

    # ------------------------------------------------------------------
    # Dependency coverage flags
    # ------------------------------------------------------------------
    st.subheader("Possible dropped dependencies")
    dep_flags = report["dependency_flags"]
    if not dep_flags:
        st.success("Matched entities preserve their call relationships.")
    else:
        st.caption("Entities present in both, but whose modernized version calls "
                   "fewer things than the legacy version — a possible dropped behavior.")
        for f in dep_flags[:40]:
            with st.expander(
                f"`{f['legacy_entity']}` → missing {len(f['missing_calls'])} call(s)"):
                st.markdown(f"**Legacy** made {f['legacy_call_count']} calls; "
                            f"**modernized** makes {f['modern_call_count']}.")
                st.markdown("**Calls not found in modernized version:** "
                            + ", ".join(f"`{c}`" for c in f["missing_calls"]))
                st.caption(f"match basis: {f['match']}")

    st.divider()

    # ------------------------------------------------------------------
    # Metric comparison
    # ------------------------------------------------------------------
    st.subheader("Metric comparison")
    met = report["metrics"]
    leg, mod = met["legacy"], met["modern"]
    table = [
        {"Metric": "Total entities", "Legacy": leg["entities"], "Modernized": mod["entities"]},
        {"Metric": "Untested entities", "Legacy": leg["untested"], "Modernized": mod["untested"]},
        {"Metric": "High-risk", "Legacy": leg["bands"]["high"], "Modernized": mod["bands"]["high"]},
        {"Metric": "Medium-risk", "Legacy": leg["bands"]["medium"], "Modernized": mod["bands"]["medium"]},
        {"Metric": "Low-risk", "Legacy": leg["bands"]["low"], "Modernized": mod["bands"]["low"]},
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(
        "Coverage and dependency diffs are computed deterministically from the "
        "two knowledge graphs. The renamed-vs-dropped judgment is AI-assisted "
        "and should be reviewed."
    )