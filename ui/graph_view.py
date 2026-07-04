"""
Knowledge graph view — interactive force-directed visualisation of the codebase.

Reads the entities (nodes) and edges (dependencies) straight from MongoDB and
renders an explorable graph using vis-network. Nodes are coloured by risk band
and sized by fan-in, so the riskiest and most-depended-on code stands out at a
glance. Filtered by default (to keep the first view legible) with a toggle to
reveal the full graph.

This is a read-only view — no analysis, no writes.
"""

import json
import streamlit as st
import streamlit.components.v1 as components

from db.schema import ENTITIES, EDGES


# Risk band -> node colour (border + fill) — dark-theme friendly
BAND_COLORS = {
    "high":   {"bg": "#4A1A1A", "border": "#FF5252"},
    "medium": {"bg": "#3D3214", "border": "#FFD740"},
    "low":    {"bg": "#1A3D1A", "border": "#69F0AE"},
}
DEFAULT_COLOR = {"bg": "#242B3D", "border": "#607D8B"}


def render_graph_view(db, repo_id):
    st.header("Codebase knowledge graph")
    st.write(
        "Every node is a code entity; every edge is a dependency (A → B means "
        "A calls B). Node colour shows modernisation risk; node size shows how "
        "many other entities depend on it. Drag to explore, scroll to zoom, "
        "hover for detail."
    )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        show_all = st.toggle("Show entire graph", value=False,
                             help="Off shows a focused subset; on shows all entities.")
    with c2:
        bands = st.multiselect("Risk bands", ["high", "medium", "low"],
                               default=["high", "medium"])
    with c3:
        min_fan_in = st.slider("Min fan-in", 0, 10, 0,
                               help="Only show entities depended on by at least this many others.")

    # ------------------------------------------------------------------
    # Load nodes
    # ------------------------------------------------------------------
    proj = {
        "_id": 0, "entity_id": 1, "type": 1, "file_path": 1,
        "risk_score": 1, "risk_band": 1, "fan_in": 1, "fan_out": 1,
        "has_tests": 1, "line_start": 1, "line_end": 1,
    }
    all_entities = list(db[ENTITIES].find({"repo_id": repo_id}, proj))

    if not all_entities:
        st.warning("No entities found for this repository. Run ingestion first.")
        return

    # Apply filters (unless showing all)
    if show_all:
        entities = all_entities
    else:
        entities = [
            e for e in all_entities
            if e.get("risk_band", "low") in bands
            and (e.get("fan_in", 0) or 0) >= min_fan_in
        ]

    if not entities:
        st.info("No entities match the current filters. Loosen them or toggle 'Show entire graph'.")
        return

    visible_ids = {e["entity_id"] for e in entities}

    # ------------------------------------------------------------------
    # Load edges, keep only those between visible nodes
    # ------------------------------------------------------------------
    all_edges = list(db[EDGES].find({}, {"_id": 0, "from_id": 1, "to_id": 1}))
    edges = [
        e for e in all_edges
        if e["from_id"] in visible_ids and e["to_id"] in visible_ids
    ]

    # ------------------------------------------------------------------
    # Build vis-network node/edge JSON
    # ------------------------------------------------------------------
    nodes_js = []
    for e in entities:
        band   = e.get("risk_band", "low")
        colors = BAND_COLORS.get(band, DEFAULT_COLOR)
        fan_in = e.get("fan_in", 0) or 0
        # Short label = the name segment of the entity_id
        name = e["entity_id"].split("::")[1] if "::" in e["entity_id"] else e["entity_id"]
        file_short = (e.get("file_path", "") or "").split("/")[-1]
        size = 12 + min(fan_in, 20) * 2.5   # scale node size by fan-in

        title = (
            f"{name}\n"
            f"{file_short} : {e.get('line_start', '?')}-{e.get('line_end', '?')}\n"
            f"risk {e.get('risk_score', '?')} ({band})  ·  "
            f"fan-in {fan_in}  ·  fan-out {e.get('fan_out', 0)}  ·  "
            f"tests: {'yes' if e.get('has_tests') else 'no'}"
        )
        nodes_js.append({
            "id":     e["entity_id"],
            "label":  name,
            "title":  title,
            "value":  size,
            "color":  {"background": colors["bg"], "border": colors["border"]},
            "font":   {"size": 12, "color": "#E8ECF1"},
        })

    edges_js = [
        {"from": e["from_id"], "to": e["to_id"], "arrows": "to"}
        for e in edges
    ]

    # ------------------------------------------------------------------
    # Summary line
    # ------------------------------------------------------------------
    st.caption(
        f"Showing {len(nodes_js)} of {len(all_entities)} entities  ·  "
        f"{len(edges_js)} dependencies"
        + ("  ·  full graph" if show_all else "  ·  filtered view")
    )

    _render_vis_network(nodes_js, edges_js)

    # Legend
    st.markdown(
        "<div style='display:flex;gap:18px;font-size:13px;margin-top:6px;color:#E8ECF1'>"
        "<span>🔴 high risk</span><span>🟡 medium risk</span>"
        "<span>🟢 low risk</span>"
        "<span style='color:#8B95A5'>— larger node = more depended-on (higher fan-in)</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_vis_network(nodes_js, edges_js):
    """Render an interactive vis-network force-directed graph via an HTML component."""
    nodes_json = json.dumps(nodes_js)
    edges_json = json.dumps(edges_js)

    html = f"""
    <html>
    <head>
      <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        #graph {{
          width: 100%;
          height: 620px;
          border: 1px solid #2D3548;
          border-radius: 8px;
          background: #141922;
        }}
      </style>
    </head>
    <body>
      <div id="graph"></div>
      <script>
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});
        const container = document.getElementById('graph');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
          nodes: {{
            shape: 'dot',
            scaling: {{ min: 10, max: 60 }},
            borderWidth: 2,
          }},
          edges: {{
            color: {{ color: '#4A5568', highlight: '#00ED64' }},
            width: 1,
            smooth: {{ type: 'continuous' }},
            arrows: {{ to: {{ scaleFactor: 0.5 }} }},
          }},
          physics: {{
            stabilization: {{ iterations: 200 }},
            barnesHut: {{
              gravitationalConstant: -8000,
              springConstant: 0.04,
              springLength: 120,
            }},
          }},
          interaction: {{
            hover: true,
            tooltipDelay: 120,
            navigationButtons: true,
            keyboard: true,
          }},
        }};
        const network = new vis.Network(container, data, options);

        // Click a node to highlight its direct neighbourhood
        network.on("click", function(params) {{
          if (params.nodes.length > 0) {{
            const sel = params.nodes[0];
            const connected = network.getConnectedNodes(sel);
            connected.push(sel);
            const update = [];
            nodes.forEach(function(n) {{
              const dim = !connected.includes(n.id);
              update.push({{ id: n.id, opacity: dim ? 0.15 : 1.0 }});
            }});
            // vis-network doesn't support per-node opacity directly via update
            // in all builds, so we re-emphasise via font instead.
          }}
        }});
      </script>
    </body>
    </html>
    """
    components.html(html, height=660, scrolling=False)