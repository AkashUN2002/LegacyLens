"""
LegacyLens — Streamlit entry point.

Run with:
    streamlit run ui/app.py

The app has two phases, gated on ingestion status:
    1. Ingestion view  — upload a repo, run P1, watch progress
    2. Chat view       — talk to the ingested codebase (P2 per message)

Phase is tracked in MongoDB metadata (survives browser refresh) and
mirrored in st.session_state for snappy in-session routing.
"""

import sys
import os

# Make the project root importable when run via `streamlit run ui/app.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env BEFORE importing anything that reads environment variables
# (db.client reads MONGO_URI, agents read AWS + Voyage keys).
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import streamlit as st

from db.client import get_db
from db.schema import setup_all, is_repo_ingested, get_repo_metadata
from ui.ingestion_view import render_ingestion_view
from ui.chat_view import render_chat_view
from ui.graph_view import render_graph_view
from ui.observability_view import render_observability_view
from ui.compare_view import render_compare_view
from ui.artifacts_view import render_artifacts_view
from ui.theme import apply_theme, render_brand_header


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LegacyLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# One-time resource setup (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def init_db():
    """Connect to MongoDB and ensure indexes once per server process."""
    db = get_db()
    setup_all(db)
    from agents.memory import ensure_conversation_indexes
    ensure_conversation_indexes(db)
    from agents.observability import ensure_usage_indexes
    ensure_usage_indexes(db)
    return db


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "active_repo_id":  None,    # repo_id currently loaded in the UI
        "phase":           "ingestion",   # "ingestion" | "chat"
        "chat_history":    [],      # list of {role, content, meta}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(db):
    with st.sidebar:
        st.title("🔍 LegacyLens")
        st.caption("Talk to your legacy codebase")
        st.divider()

        repo_id = st.session_state.active_repo_id
        if repo_id and is_repo_ingested(db, repo_id):
            meta = get_repo_metadata(db, repo_id) or {}
            st.subheader("Active repository")
            st.text(meta.get("repo_name") or os.path.basename(meta.get("repo_path", "unknown")))

            col1, col2 = st.columns(2)
            col1.metric("Entities", meta.get("entity_count", 0))
            col2.metric("Edges", meta.get("edge_count", 0))
            st.metric("High-risk modules", meta.get("high_risk_count", 0))

            st.divider()
            if st.button("Analyse a different repo", use_container_width=True):
                st.session_state.active_repo_id = None
                st.session_state.phase = "ingestion"
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No repository ingested yet. Upload one to begin.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    apply_theme()           
    db = init_db()

    render_brand_header() 
    render_sidebar(db)

    # Route based on phase
    if st.session_state.phase == "chat" and st.session_state.active_repo_id:
        repo_id = st.session_state.active_repo_id
        chat_tab, graph_tab, obs_tab, cmp_tab, art_tab = st.tabs(
            ["💬 Chat", "🕸️ Knowledge graph", "📊 Observability",
             "🔀 Migration coverage", "📄 Deliverables"])
        with chat_tab:
            render_chat_view(db, repo_id)
        with graph_tab:
            render_graph_view(db, repo_id)
        with obs_tab:
            render_observability_view(db, repo_id)
        with cmp_tab:
            render_compare_view(db)
        with art_tab:
            render_artifacts_view(db, repo_id)
    else:
        render_ingestion_view(db)


if __name__ == "__main__":
    main()