from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osint_env.data.generator import PlatformViews
from osint_env.domain.models import CanonicalGraph, Edge, TaskInstance
from osint_env.env.environment import OSINTEnvironment


def _safe_label(value: str, fallback: str) -> str:
    text = str(value).strip()
    return text if text else fallback


def _canonical_graph_payload(graph: CanonicalGraph) -> dict[str, Any]:
  nodes = []
  for node in graph.nodes.values():
    attrs = node.attrs or {}
    title = "\\n".join(f"{k}: {v}" for k, v in attrs.items())
    label = _safe_label(str(attrs.get("name") or attrs.get("handle") or node.node_id), node.node_id)
    nodes.append(
      {
        "id": node.node_id,
        "label": label,
        "group": str(node.node_type.value),
        "title": title,
        "attrs": attrs,
      }
    )

  edges = []
  for idx, edge in enumerate(graph.edges):
    edges.append(
      {
        "id": f"c_{idx}",
        "from": edge.src,
        "to": edge.dst,
        "label": edge.rel,
        "arrows": "to",
        "color": "#1f2937",
        "width": 1,
        "confidence": float(edge.confidence),
        "status": "canonical",
      }
    )
  return {"nodes": nodes, "edges": edges}


def _edge_key(edge: Edge) -> tuple[str, str, str]:
    return (edge.src, edge.rel, edge.dst)


def _episode_graph_payload(pred_edges: list[Edge], truth_edges: list[Edge], graph: CanonicalGraph) -> dict[str, Any]:
    pred = {_edge_key(e): e for e in pred_edges}
    truth = {_edge_key(e): e for e in truth_edges}

    all_nodes = set()
    all_keys = set(pred) | set(truth)
    for src, _, dst in all_keys:
        all_nodes.add(src)
        all_nodes.add(dst)

    nodes = []
    for node_id in sorted(all_nodes):
        node = graph.nodes.get(node_id)
        if node is None:
            nodes.append({"id": node_id, "label": node_id, "group": "episode", "attrs": {}})
            continue
        attrs = node.attrs or {}
        label = _safe_label(str(attrs.get("name") or attrs.get("handle") or node_id), node_id)
        nodes.append({"id": node_id, "label": label, "group": str(node.node_type.value), "attrs": attrs})

    edges = []
    for idx, key in enumerate(sorted(all_keys)):
        src, rel, dst = key
        in_pred = key in pred
        in_truth = key in truth
        if in_pred and in_truth:
            color = "#16a34a"
            dashes = False
            status = "matched"
        elif in_pred:
            color = "#2563eb"
            dashes = False
            status = "pred_only"
        else:
            color = "#f59e0b"
            dashes = True
            status = "truth_only"
        edges.append(
            {
                "id": f"e_{idx}",
                "from": src,
                "to": dst,
                "label": rel,
                "arrows": "to",
                "color": color,
                "dashes": dashes,
                "width": 2,
                "status": status,
                "confidence": float((pred.get(key) or truth.get(key) or Edge(src, rel, dst)).confidence),
            }
        )

    return {"nodes": nodes, "edges": edges}


def _views_payload(views: PlatformViews) -> dict[str, Any]:
    return {
        "microblog_posts": views.microblog_posts,
        "forum_threads": views.forum_threads,
        "profiles": views.profiles,
    }


def _leaderboard_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda r: float(r.get("metrics", {}).get("leaderboard_score", 0.0)), reverse=True)
    return ranked[:200]


def export_dashboard(
    env: OSINTEnvironment,
    evaluation: dict[str, Any],
    leaderboard_records: list[dict[str, Any]],
    output_path: str,
) -> str:
    summary = evaluation.get("summary", evaluation)
    episodes = evaluation.get("episodes", [])

    task: TaskInstance | None = env.state.task if env.state else None
    truth_edges = task.supporting_edges if task else []
    pred_edges = env.memory_graph.edges if env.state else []

    payload = {
        "summary": summary,
        "episodes": episodes,
        "leaderboard": _leaderboard_payload(leaderboard_records),
        "canonical_graph": _canonical_graph_payload(env.graph),
        "episode_graph": _episode_graph_payload(pred_edges, truth_edges, env.graph),
        "views": _views_payload(env.views),
        "task": {
            "task_id": task.task_id if task else "n/a",
            "task_type": task.task_type if task else "n/a",
            "question": task.question if task else "n/a",
            "answer": task.answer if task else "n/a",
        },
    }

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>OSINT Environment Dashboard</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
  <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap\" rel=\"stylesheet\" />
  <link href=\"https://unpkg.com/vis-network@9.1.9/styles/vis-network.min.css\" rel=\"stylesheet\" />
  <script src=\"https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js\"></script>
  <style>
    :root {{
      --ink: #1d232f;
      --muted: #5f6d7a;
      --line: #d5dfe8;
      --bg: #f5f8fb;
      --card: #ffffff;
      --brand: #0f766e;
      --brand-soft: #d4f4ef;
      --accent: #d97706;
      --accent-soft: #ffe7c2;
      --ok: #15803d;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: \"Space Grotesk\", \"Segoe UI\", sans-serif;
      background:
        radial-gradient(1200px 500px at -5% -20%, #d8efe9, transparent 70%),
        radial-gradient(900px 500px at 110% -10%, #ffe9cf, transparent 65%),
        var(--bg);
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 20px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(24, 39, 59, 0.06);
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2.1fr 1fr;
      gap: 14px;
      margin-bottom: 14px;
    }}
    .hero-main {{
      background: linear-gradient(145deg, #f7fffd, #fff8ef);
      border: 1px solid #e6efe8;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.02em; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; letter-spacing: -0.01em; }}
    .muted {{ color: var(--muted); }}
    .pill-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
    .pill {{
      border: 1px solid #dce8e6;
      background: #fbfffe;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      color: #214742;
    }}
    .stats {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 10px; margin-top: 10px; }}
    .stat {{
      border: 1px dashed #cde2df;
      background: linear-gradient(180deg, #fcfffe, #f6fffc);
      border-radius: 12px;
      padding: 10px;
    }}
    .stat .k {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
    .stat .v {{ font-size: 22px; font-weight: 700; }}
    .layout {{ display: grid; grid-template-columns: 1.2fr 3fr 1.2fr; gap: 14px; margin-bottom: 14px; }}
    .control-col {{ display: flex; flex-direction: column; gap: 14px; }}
    .control-grid {{ display: grid; gap: 8px; }}
    .graph-wrap {{ position: relative; overflow: hidden; }}
    .graph {{ height: 540px; border: 1px solid var(--line); border-radius: 14px; background: #fbfdff; }}
    .graph-banner {{
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(255,255,255,0.93);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 6px 10px;
      font-size: 12px;
      z-index: 2;
      backdrop-filter: blur(4px);
    }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; }}
    .dot {{ width: 9px; height: 9px; border-radius: 999px; display: inline-block; margin-right: 4px; }}
    .mono {{ font-family: \"IBM Plex Mono\", monospace; font-size: 12px; }}
    .inline {{ display: flex; gap: 8px; align-items: center; }}
    .split {{ display: grid; grid-template-columns: 2fr 1.3fr; gap: 14px; margin-bottom: 14px; }}
    .db-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
    .tab {{
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 5px 10px;
      background: #fff;
      cursor: pointer;
      font-size: 12px;
    }}
    .tab.active {{ background: var(--brand-soft); border-color: #b5e7de; color: #08554e; }}
    .table-wrap {{ max-height: 320px; overflow: auto; border: 1px solid var(--line); border-radius: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #edf2f7; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f7fbff; z-index: 1; }}
    tr:hover td {{ background: #f9fcff; }}
    .json-view {{
      height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #0f172a;
      color: #d2f8ee;
      padding: 10px;
      margin: 0;
    }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }}
    .chart-box {{ height: 300px; }}
    select, input[type=\"search\"], button {{
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 8px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    button {{ cursor: pointer; background: #fff; }}
    button.primary {{ background: var(--brand); border-color: #0e6f68; color: #fff; }}
    .subtle {{ background: #f7fafc; }}
    @media (max-width: 1100px) {{
      .hero, .layout, .split, .charts {{ grid-template-columns: 1fr; }}
      .graph {{ height: 440px; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <section class=\"card hero-main\">
        <h1>OSINT Benchmark Dashboard</h1>
        <p class=\"muted\">Interactive explorer for canonical knowledge graph, episode traces, source platform records, and benchmark ranking.</p>
        <div class=\"pill-row\" id=\"hero-pills\"></div>
        <div class=\"stats\" id=\"stats\"></div>
      </section>
      <section class=\"card\">
        <h2>Latest Task Snapshot</h2>
        <div><strong>Task ID:</strong> <span id=\"task-id\"></span></div>
        <div><strong>Task Type:</strong> <span id=\"task-type\"></span></div>
        <div style=\"margin-top:8px\"><strong>Question</strong></div>
        <div id=\"task-question\" class=\"muted\"></div>
        <div style=\"margin-top:8px\"><strong>Answer</strong>: <span id=\"task-answer\"></span></div>
      </section>
    </div>

    <div class=\"layout\">
      <section class=\"card control-col\">
        <div>
          <h2>Graph Controls</h2>
          <div class=\"control-grid\">
            <label class=\"mono\" for=\"graph-mode\">Graph Layer</label>
            <select id=\"graph-mode\">
              <option value=\"canonical\">Canonical Graph</option>
              <option value=\"episode\">Episode Graph</option>
            </select>
            <label class=\"mono\" for=\"graph-search\">Node Search</label>
            <input id=\"graph-search\" type=\"search\" placeholder=\"Type node id or label...\" />
            <label class=\"mono\" for=\"relation-filter\">Relation Filter</label>
            <input id=\"relation-filter\" type=\"search\" placeholder=\"Filter edge labels...\" />
            <button id=\"fit-graph\" class=\"primary\">Fit Graph</button>
          </div>
        </div>
        <div>
          <h2>Node Types</h2>
          <div id=\"type-filters\" class=\"control-grid mono\"></div>
        </div>
      </section>

      <section class=\"card\">
        <h2>Graph Explorer</h2>
        <div class=\"graph-wrap\">
          <div class=\"graph-banner\" id=\"graph-banner\">Layer: Canonical Graph</div>
          <div id=\"graph-canvas\" class=\"graph\"></div>
        </div>
        <div class=\"legend\">
          <span><span class=\"dot\" style=\"background:#16a34a\"></span>matched edge</span>
          <span><span class=\"dot\" style=\"background:#2563eb\"></span>predicted only</span>
          <span><span class=\"dot\" style=\"background:#f59e0b\"></span>truth only</span>
        </div>
      </section>

      <section class=\"card control-col\">
        <div>
          <h2>Node Inspector</h2>
          <pre id=\"node-detail\" class=\"json-view\">Click a node to inspect attributes and neighbors.</pre>
        </div>
        <div>
          <h2>Edge Inspector</h2>
          <pre id=\"edge-detail\" class=\"json-view\">Click an edge to inspect relation details.</pre>
        </div>
      </section>
    </div>

    <div class=\"split\">
      <section class=\"card\">
        <h2>Original Database Explorer</h2>
        <div class=\"db-tabs\" id=\"db-tabs\"></div>
        <div class=\"inline\" style=\"margin-bottom:8px\">
          <input id=\"db-search\" type=\"search\" placeholder=\"Search records...\" style=\"flex:1\" />
          <select id=\"db-limit\">
            <option value=\"200\">200</option>
            <option value=\"500\">500</option>
            <option value=\"1000\">1000</option>
          </select>
        </div>
        <div class=\"table-wrap\"><table id=\"db-table\"></table></div>
      </section>

      <section class=\"card\">
        <h2>Selected Source Record</h2>
        <pre id=\"db-detail\" class=\"json-view\">Click a row in the database table to inspect full JSON.</pre>
      </section>
    </div>

    <div class=\"charts\">
      <section class=\"card\">
        <h2>Benchmark Summary Radar</h2>
        <div class=\"chart-box\"><canvas id=\"summary-chart\"></canvas></div>
      </section>
      <section class=\"card\">
        <h2>Episode Reward and Graph F1</h2>
        <div class=\"chart-box\"><canvas id=\"trace-chart\"></canvas></div>
      </section>
    </div>

    <section class=\"card\">
      <h2>Benchmark Leaderboard</h2>
      <div class=\"inline\" style=\"margin-bottom:8px\">
        <label class=\"mono\" for=\"leader-sort\">Sort by</label>
        <select id=\"leader-sort\" class=\"subtle\">
          <option value=\"leaderboard_score\">leaderboard_score</option>
          <option value=\"task_success_rate\">task_success_rate</option>
          <option value=\"avg_graph_f1\">avg_graph_f1</option>
          <option value=\"retrieval_signal\">retrieval_signal</option>
          <option value=\"structural_signal\">structural_signal</option>
          <option value=\"spawn_signal\">spawn_signal</option>
          <option value=\"avg_reward\">avg_reward</option>
        </select>
      </div>
      <div class=\"table-wrap\"><table id=\"leaderboard-table\"></table></div>
    </section>
  </div>

  <script>
    const payload = {json.dumps(payload)};

    function metricCards(summary) {{
      const selected = [
        ["leaderboard_score", summary.leaderboard_score || 0],
        ["task_success_rate", summary.task_success_rate || 0],
        ["avg_graph_f1", summary.avg_graph_f1 || 0],
        ["retrieval_signal", summary.retrieval_signal || 0],
        ["structural_signal", summary.structural_signal || 0],
        ["tool_efficiency", summary.tool_efficiency || 0],
        ["avg_reward", summary.avg_reward || 0]
      ];
      const root = document.getElementById("stats");
      root.innerHTML = "";
      selected.forEach(([k, v]) => {{
        const div = document.createElement("div");
        div.className = "stat";
        div.innerHTML = `<div class=\"k\">${{k}}</div><div class=\"v\">${{Number(v).toFixed(3)}}</div>`;
        root.appendChild(div);
      }});

      const pillRow = document.getElementById("hero-pills");
      pillRow.innerHTML = "";
      [
        `deanonymization: ${{Number(summary.deanonymization_accuracy || 0).toFixed(3)}}`,
        `avg steps: ${{Number(summary.avg_steps_to_solution || 0).toFixed(2)}}`,
        `episodes: ${{(payload.episodes || []).length}}`
      ].forEach((text) => {{
        const span = document.createElement("span");
        span.className = "pill";
        span.textContent = text;
        pillRow.appendChild(span);
      }});
    }}

    function buildTypeFilters(allGroups) {{
      const root = document.getElementById("type-filters");
      root.innerHTML = "";
      allGroups.forEach((group) => {{
        const id = `type_${{group}}`;
        const row = document.createElement("label");
        row.className = "inline";
        row.innerHTML = `<input type=\"checkbox\" id=\"${{id}}\" value=\"${{group}}\" checked /> <span>${{group}}</span>`;
        root.appendChild(row);
      }});
    }}

    function createNetworkController() {{
      const container = document.getElementById("graph-canvas");
      const banner = document.getElementById("graph-banner");
      const modeSelect = document.getElementById("graph-mode");
      const nodeSearch = document.getElementById("graph-search");
      const relFilter = document.getElementById("relation-filter");
      const fitBtn = document.getElementById("fit-graph");

      const rawLayers = {{
        canonical: payload.canonical_graph || {{ nodes: [], edges: [] }},
        episode: payload.episode_graph || {{ nodes: [], edges: [] }}
      }};

      const allGroups = Array.from(new Set((rawLayers.canonical.nodes || []).map(n => n.group || "unknown"))).sort();
      buildTypeFilters(allGroups);

      const state = {{
        mode: "canonical",
        relationQuery: "",
        nodeQuery: "",
      }};

      const nodesDS = new vis.DataSet([]);
      const edgesDS = new vis.DataSet([]);
      const network = new vis.Network(container, {{ nodes: nodesDS, edges: edgesDS }}, {{
        interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
        physics: {{ stabilization: false, barnesHut: {{ springLength: 130 }} }},
        edges: {{ smooth: true, font: {{ size: 10 }} }},
        nodes: {{ shape: "dot", size: 11, font: {{ size: 10 }} }}
      }});

      function activeGroups() {{
        const checked = Array.from(document.querySelectorAll('#type-filters input[type="checkbox"]:checked'));
        return new Set(checked.map(x => x.value));
      }}

      function styleNode(node, query) {{
        const text = `${{node.id}} ${{node.label || ""}}`.toLowerCase();
        const hit = query && text.includes(query);
        return {{
          ...node,
          color: hit ? "#f59e0b" : undefined,
          size: hit ? 18 : 11,
        }};
      }}

      function refresh() {{
        const raw = rawLayers[state.mode] || {{ nodes: [], edges: [] }};
        const groups = activeGroups();
        const relQ = state.relationQuery.toLowerCase();
        const nodeQ = state.nodeQuery.toLowerCase();

        const nodes = (raw.nodes || []).filter(n => groups.has(n.group || "unknown")).map(n => styleNode(n, nodeQ));
        const nodeIds = new Set(nodes.map(n => n.id));
        const edges = (raw.edges || []).filter(e => nodeIds.has(e.from) && nodeIds.has(e.to)).filter(e => !relQ || String(e.label || "").toLowerCase().includes(relQ));

        nodesDS.clear();
        edgesDS.clear();
        nodesDS.add(nodes);
        edgesDS.add(edges);

        banner.textContent = state.mode === "canonical" ? "Layer: Canonical Graph" : "Layer: Episode Graph";
      }}

      modeSelect.addEventListener("change", () => {{
        state.mode = modeSelect.value;
        refresh();
      }});
      relFilter.addEventListener("input", () => {{
        state.relationQuery = relFilter.value || "";
        refresh();
      }});
      nodeSearch.addEventListener("input", () => {{
        state.nodeQuery = nodeSearch.value || "";
        refresh();
      }});
      fitBtn.addEventListener("click", () => network.fit({{ animation: true }}));
      document.getElementById("type-filters").addEventListener("change", refresh);

      network.on("click", (params) => {{
        if (params.nodes && params.nodes.length) {{
          const node = nodesDS.get(params.nodes[0]);
          const connected = network.getConnectedNodes(node.id) || [];
          document.getElementById("node-detail").textContent = JSON.stringify({{
            node,
            connected_nodes: connected
          }}, null, 2);
        }}
        if (params.edges && params.edges.length) {{
          const edge = edgesDS.get(params.edges[0]);
          document.getElementById("edge-detail").textContent = JSON.stringify(edge, null, 2);
        }}
      }});

      refresh();
    }}

    function buildRows(views) {{
      const rows = [];
      (views.microblog_posts || []).forEach((x) => rows.push({{ source: "microblog", id: x.post_id || "post", text: JSON.stringify(x), raw: x }}));
      (views.forum_threads || []).forEach((x) => rows.push({{ source: "forum", id: x.thread_id || "thread", text: JSON.stringify(x), raw: x }}));
      (views.profiles || []).forEach((x) => rows.push({{ source: "profile", id: x.user_id || "profile", text: JSON.stringify(x), raw: x }}));
      return rows;
    }}

    function initDatabaseExplorer() {{
      const rows = buildRows(payload.views || {{}});
      const tabs = document.getElementById("db-tabs");
      const search = document.getElementById("db-search");
      const limit = document.getElementById("db-limit");
      const table = document.getElementById("db-table");
      const detail = document.getElementById("db-detail");

      const sources = ["all", "microblog", "forum", "profile"];
      const state = {{ source: "all", query: "", limit: 200 }};

      tabs.innerHTML = "";
      sources.forEach((src) => {{
        const btn = document.createElement("button");
        btn.className = `tab ${{src === state.source ? "active" : ""}}`;
        btn.textContent = src;
        btn.addEventListener("click", () => {{
          state.source = src;
          Array.from(tabs.children).forEach((child) => child.classList.remove("active"));
          btn.classList.add("active");
          render();
        }});
        tabs.appendChild(btn);
      }});

      function filtered() {{
        const q = state.query.toLowerCase();
        return rows
          .filter((row) => state.source === "all" || row.source === state.source)
          .filter((row) => !q || row.text.toLowerCase().includes(q) || row.id.toLowerCase().includes(q));
      }}

      function render() {{
        const show = filtered().slice(0, state.limit);
        table.innerHTML = "<thead><tr><th>source</th><th>id</th><th>preview</th></tr></thead>";
        const body = document.createElement("tbody");
        show.forEach((row) => {{
          const tr = document.createElement("tr");
          const preview = row.text.length > 120 ? `${{row.text.slice(0, 120)}}...` : row.text;
          tr.innerHTML = `<td>${{row.source}}</td><td class=\"mono\">${{row.id}}</td><td>${{preview}}</td>`;
          tr.addEventListener("click", () => {{
            detail.textContent = JSON.stringify(row.raw, null, 2);
          }});
          body.appendChild(tr);
        }});
        table.appendChild(body);
      }}

      search.addEventListener("input", () => {{ state.query = search.value || ""; render(); }});
      limit.addEventListener("change", () => {{ state.limit = Number(limit.value || 200); render(); }});
      render();
    }}

    function renderLeaderboard(records, sortBy = "leaderboard_score") {{
      const sorted = [...records].sort((a, b) => (b.metrics?.[sortBy] || 0) - (a.metrics?.[sortBy] || 0));
      const table = document.getElementById("leaderboard-table");
      table.innerHTML = "<thead><tr><th>rank</th><th>run</th><th>score</th><th>success</th><th>graph_f1</th><th>retrieval</th><th>structural</th><th>spawn</th><th>reward</th></tr></thead>";
      const body = document.createElement("tbody");
      sorted.forEach((rec, i) => {{
        const m = rec.metrics || {{}};
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{i + 1}}</td><td>${{rec.run_name || rec.run_id || "run"}}</td><td>${{(m.leaderboard_score || 0).toFixed(4)}}</td><td>${{(m.task_success_rate || 0).toFixed(3)}}</td><td>${{(m.avg_graph_f1 || 0).toFixed(3)}}</td><td>${{(m.retrieval_signal || 0).toFixed(3)}}</td><td>${{(m.structural_signal || 0).toFixed(3)}}</td><td>${{(m.spawn_signal || 0).toFixed(3)}}</td><td>${{(m.avg_reward || 0).toFixed(3)}}</td>`;
        body.appendChild(tr);
      }});
      table.appendChild(body);
    }}

    function drawSummaryChart(summary) {{
      const labels = ["success", "graph_f1", "tool_eff", "deanon", "retrieval", "structural", "score"];
      const values = [
        summary.task_success_rate || 0,
        summary.avg_graph_f1 || 0,
        summary.tool_efficiency || 0,
        summary.deanonymization_accuracy || 0,
        summary.retrieval_signal || 0,
        summary.structural_signal || 0,
        summary.leaderboard_score || 0,
      ];
      new Chart(document.getElementById("summary-chart"), {{
        type: "radar",
        data: {{
          labels,
          datasets: [{{
            label: "normalized metrics",
            data: values,
            backgroundColor: "rgba(15,118,110,0.2)",
            borderColor: "#0f766e",
            pointBackgroundColor: "#d97706",
            pointRadius: 3
          }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ r: {{ min: 0, max: 1 }} }} }}
      }});
    }}

    function drawTraceChart(episodes) {{
      const labels = episodes.map((_, i) => `ep_${{i + 1}}`);
      const rewards = episodes.map(e => e.reward || 0);
      const f1 = episodes.map(e => e.graph_f1 || 0);
      new Chart(document.getElementById("trace-chart"), {{
        type: "line",
        data: {{
          labels,
          datasets: [
            {{ label: "reward", data: rewards, borderColor: "#0f766e", yAxisID: "y", tension: 0.2 }},
            {{ label: "graph_f1", data: f1, borderColor: "#d97706", yAxisID: "y1", tension: 0.2 }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            y: {{ position: "left" }},
            y1: {{ position: "right", min: 0, max: 1, grid: {{ drawOnChartArea: false }} }}
          }}
        }}
      }});
    }}

    const summary = payload.summary || {{}};
    metricCards(summary);

    document.getElementById("task-id").textContent = payload.task.task_id;
    document.getElementById("task-type").textContent = payload.task.task_type;
    document.getElementById("task-question").textContent = payload.task.question;
    document.getElementById("task-answer").textContent = payload.task.answer;

    createNetworkController();
    initDatabaseExplorer();

    const leaderboard = payload.leaderboard || [];
    const leaderSort = document.getElementById("leader-sort");
    renderLeaderboard(leaderboard, leaderSort.value);
    leaderSort.addEventListener("change", () => renderLeaderboard(leaderboard, leaderSort.value));

    drawSummaryChart(summary);
    drawTraceChart(payload.episodes || []);
  </script>
</body>
</html>
"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)
