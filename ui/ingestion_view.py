"""
Ingestion view (Phase 1).

Lets the user upload a codebase .zip and runs the ingestion pipeline.
Shows progress through each agent so judges see the system working, then
a summary card that transitions into the chat phase.
"""

import os
import zipfile
import tempfile
import streamlit as st

from db.collections import is_repo_ingested, get_repo_metadata, METADATA
from agents.parser_agent import _repo_id   # reuse the same hashing logic
from graph.ingestion_pipeline import build_ingestion_pipeline
from db.checkpointer import get_checkpointer, clear_checkpoints

EXTRACT_BASE = os.path.join(tempfile.gettempdir(), "legacylens_uploads")


def _extract_zip(uploaded_file):
    """
    Extract an uploaded .zip into a stable temp directory.
    Returns (repo_path, repo_name).
    """
    zip_stem = os.path.splitext(uploaded_file.name)[0]
    extract_dir = os.path.join(EXTRACT_BASE, zip_stem)

    if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            # Guard against zip-slip (path traversal)
            top = os.path.realpath(extract_dir)
            for member in zf.namelist():
                dest = os.path.realpath(os.path.join(extract_dir, member))
                if not dest.startswith(top + os.sep) and dest != top:
                    raise zipfile.BadZipFile(
                        "Zip contains unsafe path traversal entries"
                    )
            zf.extractall(extract_dir)

    # If the zip has a single top-level directory, use it as the repo root
    entries = os.listdir(extract_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        return os.path.join(extract_dir, entries[0]), zip_stem
    return extract_dir, zip_stem


def render_ingestion_view(db):
    st.header("Analyse a legacy codebase")
    st.write(
        "Upload a .zip of your repository. LegacyLens will parse the code, "
        "map every dependency, score modernisation risk, and build a "
        "searchable graph — then you can ask it anything."
    )

    # ------------------------------------------------------------------
    # Repo source selection
    # ------------------------------------------------------------------
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Codebase (.zip)",
            type=["zip"],
            help="Upload a .zip file containing the codebase to analyse.",
            label_visibility="collapsed",
        )
    with col2:
        language = st.selectbox("Language", ["python", "java"], index=1)

    # ------------------------------------------------------------------
    # Pre-flight: is this repo already ingested?
    # ------------------------------------------------------------------
    if uploaded_file is not None:
        try:
            repo_path, repo_name = _extract_zip(uploaded_file)
        except (zipfile.BadZipFile, OSError) as exc:
            st.error(f"Could not extract zip: {exc}")
            return

        repo_id = _repo_id(repo_path)
        already = is_repo_ingested(db, repo_id)

        if already:
            meta = get_repo_metadata(db, repo_id) or {}
            st.success(
                f"This repo is already analysed — "
                f"{meta.get('entity_count', 0)} entities, "
                f"{meta.get('high_risk_count', 0)} high-risk modules."
            )
            c1, c2 = st.columns(2)
            if c1.button("Open chat", type="primary", use_container_width=True):
                _go_to_chat(repo_id)
            if c2.button("Re-analyse from scratch", use_container_width=True):
                _run_ingestion(db, repo_path, language, repo_id, repo_name=repo_name, force=True)
        else:
            if st.button("Analyse repository", type="primary", use_container_width=True):
                _run_ingestion(db, repo_path, language, repo_id, repo_name=repo_name)


# ---------------------------------------------------------------------------
# Run the pipeline with staged progress
# ---------------------------------------------------------------------------

def _run_ingestion(db, repo_path, language, repo_id, repo_name=None, force=False):
    if force:
        clear_checkpoints(thread_id=repo_id)

    # Staged progress UI — one line per agent
    stages = [
        ("parser",        "Parsing source · extracting entities"),
        ("graph_builder", "Mapping dependencies · enriching with AI"),
        ("risk_analyst",  "Scoring risk · building search index"),
    ]

    status_box = st.status("Running ingestion pipeline…", expanded=True)
    progress   = st.progress(0.0)

    # We stream the LangGraph execution node-by-node so each stage updates live.
    checkpointer = get_checkpointer()
    app = build_ingestion_pipeline(checkpointer=checkpointer)

    initial_state = {
        "repo_path": repo_path,
        "language":  language,
        "db":        db,
    }
    config = {"configurable": {"thread_id": repo_id}} if checkpointer else {}

    completed = 0
    final_state = {}

    try:
        # LangGraph .stream() yields after each node completes
        for chunk in app.stream(initial_state, config=config):
            for node_name, node_output in chunk.items():
                final_state.update(node_output or {})
                # Match node to its friendly label
                for idx, (sid, label) in enumerate(stages):
                    if sid == node_name:
                        status_box.write(f"✓ {label}")
                        completed = idx + 1
                        progress.progress(completed / len(stages))
        status_box.update(label="Ingestion complete", state="complete")
        if repo_name:
            db[METADATA].update_one(
                {"repo_id": repo_id},
                {"$set": {"repo_name": repo_name}},
            )
    except Exception as exc:
        status_box.update(label="Ingestion failed", state="error")
        st.error(f"Ingestion error: {exc}")
        return

    # ------------------------------------------------------------------
    # Summary card
    # ------------------------------------------------------------------
    _render_summary(db, repo_id)

    if st.button("Start chatting with this codebase", type="primary", use_container_width=True):
        _go_to_chat(repo_id)


def _render_summary(db, repo_id):
    meta = get_repo_metadata(db, repo_id) or {}
    st.subheader("Analysis complete")
    c1, c2, c3 = st.columns(3)
    c1.metric("Entities found", meta.get("entity_count", 0))
    c2.metric("Dependencies mapped", meta.get("edge_count", 0))
    c3.metric("High-risk modules", meta.get("high_risk_count", 0))

    errors = meta.get("parse_errors", [])
    if errors:
        with st.expander(f"{len(errors)} files could not be parsed"):
            for e in errors[:50]:
                st.text(e)


# ---------------------------------------------------------------------------
# Phase transition
# ---------------------------------------------------------------------------

def _go_to_chat(repo_id):
    st.session_state.active_repo_id = repo_id
    st.session_state.phase = "chat"
    st.session_state.chat_history = []
    st.rerun()