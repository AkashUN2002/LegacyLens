"""
Observability view — token, cost, and latency dashboard for a repo.

Reads the `usage` collection (populated by agents/observability.py) and shows
where tokens and time go: ingestion vs chat, broken down by operation.
"""

import streamlit as st
from agents.observability import repo_summary


def render_observability_view(db, repo_id):
    st.header("Observability")
    st.write(
        "Token usage, estimated cost, and latency for this repository — "
        "split between the one-time ingestion and ongoing chat. All metrics "
        "are recorded to MongoDB as each agent runs."
    )

    summary = repo_summary(db, repo_id)

    if summary["total"]["calls"] == 0:
        st.info("No usage recorded yet for this repository. Run an analysis or "
                "ask a question, then check back.")
        return

    # ------------------------------------------------------------------
    # Top-line totals
    # ------------------------------------------------------------------
    t = summary["total"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tokens", f"{t['tokens']:,}")
    c2.metric("Estimated cost", f"${t['cost']:.4f}")
    c3.metric("Total LLM/embed calls", f"{t['calls']:,}")
    c4.metric("Cumulative latency", f"{t['latency']:.1f}s")

    st.divider()

    # ------------------------------------------------------------------
    # Ingestion vs chat split
    # ------------------------------------------------------------------
    col_ing, col_qry = st.columns(2)
    with col_ing:
        ing = summary["ingestion"]
        st.subheader("Ingestion (one-time)")
        st.metric("Tokens", f"{ing['tokens']:,}")
        st.metric("Cost", f"${ing['cost']:.4f}")
        st.caption(f"{ing['calls']} calls · {ing['latency']:.1f}s total")
    with col_qry:
        qry = summary["query"]
        st.subheader("Chat (ongoing)")
        st.metric("Tokens", f"{qry['tokens']:,}")
        st.metric("Cost", f"${qry['cost']:.4f}")
        st.caption(f"{qry['calls']} calls · {qry['latency']:.1f}s total")

    st.divider()

    # ------------------------------------------------------------------
    # Per-operation breakdown
    # ------------------------------------------------------------------
    st.subheader("Breakdown by operation")
    rows = sorted(summary["by_operation"],
                  key=lambda r: r["total_tokens"], reverse=True)
    if rows:
        table = [
            {
                "Phase":      r["phase"],
                "Operation":  r["operation"],
                "Provider":   r["provider"],
                "Calls":      r["calls"],
                "Input tok":  f"{r['input_tokens']:,}",
                "Output tok": f"{r['output_tokens']:,}",
                "Total tok":  f"{r['total_tokens']:,}",
                "Cost $":     f"{r['cost_usd']:.5f}",
                "Latency s":  f"{r['latency_s']:.2f}",
            }
            for r in rows
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Recent query latency trend
    # ------------------------------------------------------------------
    recent = summary.get("recent_queries", [])
    if recent:
        st.subheader("Recent chat operations")
        st.caption("Most recent query-phase operations (newest last).")
        trend = [
            {
                "Operation":  r.get("operation", ""),
                "Tokens":     r.get("total_tokens", 0),
                "Latency s":  round(r.get("latency_s", 0), 2),
                "Cost $":     round(r.get("cost_usd", 0), 5),
            }
            for r in recent
        ]
        st.dataframe(trend, use_container_width=True, hide_index=True)

    st.caption(
        "Cost figures are estimates using approximate per-model rates and are "
        "for relative insight, not billing."
    )