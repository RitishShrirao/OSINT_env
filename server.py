from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.env.environment import OSINTEnvironment
from osint_env.eval.runner import run_evaluation
from osint_env.llm import build_llm_client
from osint_env.viz import export_dashboard


SPACE_CONFIG_PATH = Path(os.getenv("OSINT_ENV_CONFIG", "datasets/fixed_levels/shared_config_fixed_levels.json"))
SPACE_SEED_PATH = Path(os.getenv("OSINT_ENV_SEED_FILE", "datasets/fixed_levels/seed_fixed_levels.json"))
SPACE_PROVIDER = os.getenv("OSINT_SPACE_LLM_PROVIDER", "mock")
SPACE_MODEL = os.getenv("OSINT_SPACE_LLM_MODEL", "gpt-4o-mini")
SPACE_PORT = int(os.getenv("PORT", "7860"))
SPACE_DASHBOARD = Path("artifacts/space_dashboard.html")


def _build_environment() -> OSINTEnvironment:
    shared = load_shared_config(SPACE_CONFIG_PATH)
    env_cfg = clone_environment_config(shared.environment)
    if SPACE_SEED_PATH.exists():
        env_cfg.seeding = load_seeding_config(SPACE_SEED_PATH)
    env_cfg.llm.provider = SPACE_PROVIDER
    env_cfg.llm.model = SPACE_MODEL
    try:
        llm = build_llm_client(env_cfg.llm)
    except Exception:
        env_cfg.llm.provider = "mock"
        llm = build_llm_client(env_cfg.llm)
    return OSINTEnvironment(env_cfg, llm=llm)


@lru_cache(maxsize=1)
def _space_snapshot() -> dict[str, Any]:
    env = _build_environment()
    evaluation = run_evaluation(env, episodes=3, return_details=True, llm=build_llm_client(env.config.llm))
    dashboard_path = export_dashboard(
        env=env,
        evaluation=evaluation,
        leaderboard_records=[],
        output_path=str(SPACE_DASHBOARD),
    )
    difficulty_counts = Counter(str(task.metadata.get("difficulty", "unknown")) for task in env.tasks)
    return {
        "dashboard_path": dashboard_path,
        "summary": evaluation["summary"],
        "task_count": len(env.tasks),
        "difficulty_counts": dict(difficulty_counts),
        "action_space": ["CALL_TOOL", "ADD_EDGE", "ANSWER"],
        "observation_space": {
            "tool_outputs": "Last tool results and memory hits.",
            "graph_snapshot": "Current working graph edge snapshot.",
            "action_history": "Recent action/reward trace.",
            "task": "Task id, task type, and question.",
        },
        "task_types": sorted({task.task_type for task in env.tasks}),
        "config": {
            "seed": env.config.seed,
            "max_steps": env.config.max_steps,
            "swarm_enabled": env.config.swarm.enabled,
            "llm_provider": env.config.llm.provider,
            "llm_model": env.config.llm.model,
        },
    }


app = FastAPI(title="OSINT OpenEnv Space", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    snapshot = _space_snapshot()
    summary = snapshot["summary"]
    difficulty_html = "".join(
        f"<li><strong>{level}</strong>: {count}</li>"
        for level, count in sorted(snapshot["difficulty_counts"].items())
    )
    task_type_html = "".join(f"<li>{task_type}</li>" for task_type in snapshot["task_types"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OSINT OpenEnv Space</title>
  <style>
    :root {{
      --ink: #13212d;
      --muted: #4d5b69;
      --line: #d8e2eb;
      --card: #ffffff;
      --bg: #f6fafc;
      --brand: #0f766e;
      --accent: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(180,83,9,0.10), transparent 28%),
        var(--bg);
    }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    .hero, .grid {{ display: grid; gap: 16px; }}
    .hero {{ grid-template-columns: 1.5fr 1fr; }}
    .grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 16px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 24px rgba(19, 33, 45, 0.06);
    }}
    h1, h2 {{ margin-top: 0; }}
    .muted {{ color: var(--muted); }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .stat {{ border: 1px dashed var(--line); border-radius: 12px; padding: 10px; }}
    .stat .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase; }}
    .stat .v {{ font-size: 22px; font-weight: 700; }}
    a.button {{
      display: inline-block;
      padding: 10px 14px;
      border-radius: 12px;
      text-decoration: none;
      color: white;
      background: var(--brand);
      margin-right: 10px;
    }}
    a.link {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    ul {{ padding-left: 18px; }}
    code {{
      background: #f1f5f9;
      border-radius: 6px;
      padding: 2px 6px;
    }}
    @media (max-width: 900px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <section class="card">
        <h1>OSINT OpenEnv Space</h1>
        <p class="muted">A containerized OpenEnv-compatible benchmark for synthetic OSINT reasoning over profiles, forum threads, posts, aliases, organizations, locations, and event links.</p>
        <p>The Space boots with the fixed-level benchmark so visitors get a stable environment snapshot instead of a different graph every restart.</p>
        <a class="button" href="/dashboard">Open Dashboard</a>
        <a class="link" href="/api/environment">Environment JSON</a>
      </section>
      <section class="card">
        <h2>Included Snapshot</h2>
        <div class="stats">
          <div class="stat"><div class="k">Tasks</div><div class="v">{snapshot["task_count"]}</div></div>
          <div class="stat"><div class="k">Provider</div><div class="v">{snapshot["config"]["llm_provider"]}</div></div>
          <div class="stat"><div class="k">Score</div><div class="v">{summary["leaderboard_score"]:.3f}</div></div>
          <div class="stat"><div class="k">Success</div><div class="v">{summary["task_success_rate"]:.3f}</div></div>
        </div>
      </section>
    </div>

    <div class="grid">
      <section class="card">
        <h2>Action Space</h2>
        <ul>
          <li><code>CALL_TOOL</code>: query platform views or semantic memory.</li>
          <li><code>ADD_EDGE</code>: add a hypothesized relation to the working graph.</li>
          <li><code>ANSWER</code>: submit the final node id answer.</li>
        </ul>
      </section>
      <section class="card">
        <h2>Difficulty Mix</h2>
        <ul>{difficulty_html}</ul>
      </section>
      <section class="card">
        <h2>Task Families</h2>
        <ul>{task_type_html}</ul>
      </section>
    </div>
  </div>
</body>
</html>"""


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/environment")
def environment_metadata() -> JSONResponse:
    return JSONResponse(_space_snapshot())


@app.get("/dashboard")
def dashboard() -> FileResponse:
    snapshot = _space_snapshot()
    return FileResponse(snapshot["dashboard_path"], media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=SPACE_PORT)

