from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from osint_env.api import (
    OpenEnvActionRequest,
    OpenEnvInferenceReportRequest,
    OpenEnvInferenceReportResponse,
    OpenEnvObservationModel,
    OpenEnvResetRequest,
    OpenEnvResponseEnvelope,
    OpenEnvTaskSummary,
)
from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.domain.models import Action, ActionType
from osint_env.env.environment import OSINTEnvironment
from osint_env.eval.leaderboard import load_leaderboard
from osint_env.eval.runner import run_evaluation
from osint_env.llm import build_llm_client
from osint_env.viz import export_dashboard


SPACE_CONFIG_PATH = Path(os.getenv("OSINT_ENV_CONFIG", "datasets/fixed_levels/shared_config_fixed_levels.json"))
SPACE_SEED_PATH = Path(os.getenv("OSINT_ENV_SEED_FILE", "datasets/fixed_levels/seed_fixed_levels.json"))
SPACE_PROVIDER = os.getenv("OSINT_SPACE_LLM_PROVIDER", "mock")
SPACE_MODEL = os.getenv("OSINT_SPACE_LLM_MODEL", "gpt-4o-mini")
SPACE_PORT = int(os.getenv("PORT", "7860"))
SPACE_DASHBOARD = Path("artifacts/space_dashboard.html")
LATEST_BASELINE_OUTPUT = Path("artifacts/baselines/openai_fixed_levels_latest.json")
LATEST_EVALUATION_OUTPUT = Path("artifacts/latest_evaluation.json")
OPENENV_SPEC_PATH = Path("openenv.yaml")

_SESSION_LOCK = Lock()
_SESSIONS: dict[str, OSINTEnvironment] = {}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


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


def _serialize_observation(observation: Any) -> OpenEnvObservationModel:
    return OpenEnvObservationModel(
        tool_outputs=list(observation.tool_outputs),
        graph_snapshot=dict(observation.graph_snapshot),
        action_history=list(observation.action_history),
        task=dict(observation.task),
    )


def _safe_session_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_count": int(info.get("step_count", 0)),
        "total_reward": float(info.get("total_reward", 0.0)),
        "tool_calls": int(info.get("tool_calls", 0)),
        "redundant_tool_calls": int(info.get("redundant_tool_calls", 0)),
        "task_answer": str(info.get("task_answer", "")),
        "agent_answer": "" if info.get("agent_answer") is None else str(info.get("agent_answer", "")),
        "graph_f1": float(info.get("graph_f1", 0.0)),
        "reward_components": dict(info.get("reward_components", {})),
    }


def _task_summaries(env: OSINTEnvironment) -> list[OpenEnvTaskSummary]:
    return [
        OpenEnvTaskSummary(
            task_id=task.task_id,
            task_type=task.task_type,
            question=task.question,
            difficulty=str(task.metadata.get("difficulty", "unknown")),
        )
        for task in env.tasks
    ]


def _resolve_task_index(env: OSINTEnvironment, request: OpenEnvResetRequest) -> int:
    if request.task_index is not None:
        task_index = int(request.task_index)
        if task_index < 0 or task_index >= len(env.tasks):
            raise HTTPException(status_code=400, detail=f"Invalid task_index {task_index}")
        return task_index
    if request.task_id:
        for idx, task in enumerate(env.tasks):
            if task.task_id == request.task_id:
                return idx
        raise HTTPException(status_code=400, detail=f"Unknown task_id {request.task_id}")
    return 0


def _get_session_env(session_id: str) -> OSINTEnvironment:
    with _SESSION_LOCK:
        env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id {session_id}")
    return env


def _store_session(session_id: str, env: OSINTEnvironment) -> None:
    with _SESSION_LOCK:
        _SESSIONS[session_id] = env


def _task_lookup(env: OSINTEnvironment) -> dict[str, Any]:
    return {task.task_id: task for task in env.tasks}


def _normalize_episode_rows(env: OSINTEnvironment, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks_by_id = _task_lookup(env)
    normalized: list[dict[str, Any]] = []
    for episode in episodes:
        row = dict(episode)
        task = tasks_by_id.get(str(row.get("task_id", "")))
        if task is not None:
            row.setdefault("task_type", task.task_type)
            row.setdefault("question", task.question)
            row.setdefault("task_answer", task.answer)
            row.setdefault(
                "truth_edges",
                [
                    {
                        "src": edge.src,
                        "rel": edge.rel,
                        "dst": edge.dst,
                        "confidence": float(edge.confidence),
                    }
                    for edge in task.supporting_edges
                ],
            )
        row.setdefault("pred_edges", [])
        row.setdefault("reward_components", {})
        row.setdefault("graph_f1", 0.0)
        row.setdefault("reward", 0.0)
        row.setdefault("steps", 0)
        row.setdefault("tool_calls", 0)
        row.setdefault("success", 0)
        normalized.append(row)
    return normalized


@lru_cache(maxsize=1)
def _base_environment_snapshot() -> dict[str, Any]:
    env = _build_environment()
    difficulty_counts = Counter(str(task.metadata.get("difficulty", "unknown")) for task in env.tasks)
    return {
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


@lru_cache(maxsize=1)
def _preview_snapshot() -> dict[str, Any]:
    env = _build_environment()
    evaluation = run_evaluation(env, episodes=3, return_details=True, llm=build_llm_client(env.config.llm))
    dashboard_path = export_dashboard(
        env=env,
        evaluation=evaluation,
        leaderboard_records=[],
        output_path=str(SPACE_DASHBOARD),
    )
    snapshot = dict(_base_environment_snapshot())
    snapshot["summary"] = evaluation["summary"]
    snapshot["dashboard_path"] = dashboard_path
    return snapshot


def _space_snapshot() -> dict[str, Any]:
    snapshot = dict(_base_environment_snapshot())

    baseline_payload = _load_json(LATEST_BASELINE_OUTPUT)
    evaluation_payload = _load_json(LATEST_EVALUATION_OUTPUT)

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    if baseline_payload is not None and isinstance(baseline_payload.get("summary"), dict):
        candidates.append((_path_mtime(LATEST_BASELINE_OUTPUT), "baseline_output", baseline_payload))
    if evaluation_payload is not None and isinstance(evaluation_payload.get("summary"), dict):
        candidates.append((_path_mtime(LATEST_EVALUATION_OUTPUT), "latest_evaluation", evaluation_payload))

    if candidates:
        _, source, payload = max(candidates, key=lambda item: item[0])
        snapshot["summary"] = dict(payload["summary"])
        snapshot["source"] = source
        if source == "baseline_output":
            dashboard_path = Path(
                str(
                    ((payload.get("run") or {}).get("dashboard_path"))
                    or "artifacts/baselines/openai_fixed_levels_dashboard.html"
                )
            )
            if dashboard_path.exists():
                snapshot["dashboard_path"] = str(dashboard_path)
            return snapshot

        env = _build_environment()
        dashboard_path = export_dashboard(
            env=env,
            evaluation=payload,
            leaderboard_records=[],
            output_path=str(SPACE_DASHBOARD),
        )
        snapshot["dashboard_path"] = dashboard_path
        return snapshot

    preview = _preview_snapshot()
    preview["source"] = "preview"
    return preview


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


@app.get("/openenv.yaml")
def openenv_spec() -> FileResponse:
    return FileResponse(OPENENV_SPEC_PATH, media_type="text/yaml")


@app.get("/api/environment")
def environment_metadata() -> JSONResponse:
    return JSONResponse(_space_snapshot())


@app.get("/openenv/tasks", response_model=list[OpenEnvTaskSummary])
def openenv_tasks() -> list[OpenEnvTaskSummary]:
    env = _build_environment()
    return _task_summaries(env)


@app.post("/openenv/reset", response_model=OpenEnvResponseEnvelope)
async def openenv_reset(request: Request) -> OpenEnvResponseEnvelope:
    env = _build_environment()
    raw_body = await request.body()
    if not raw_body.strip():
        payload: dict[str, Any] = {}
    else:
        try:
            parsed_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Reset body must be valid JSON") from exc
        if parsed_payload is None:
            payload = {}
        elif isinstance(parsed_payload, dict):
            payload = parsed_payload
        else:
            raise HTTPException(status_code=400, detail="Reset body must be a JSON object")

    try:
        reset_request = OpenEnvResetRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid reset request payload") from exc

    env._task_idx = _resolve_task_index(env, reset_request)
    observation = env.reset()
    session_id = str(uuid4())
    _store_session(session_id, env)
    return OpenEnvResponseEnvelope(
        session_id=session_id,
        observation=_serialize_observation(observation),
        reward=0.0,
        done=False,
        info=_safe_session_info(env._info()),
    )


@app.post("/openenv/step", response_model=OpenEnvResponseEnvelope)
def openenv_step(request: OpenEnvActionRequest) -> OpenEnvResponseEnvelope:
    env = _get_session_env(request.session_id)
    action_type_raw = request.resolved_action_type().strip()
    if not action_type_raw:
        raise HTTPException(status_code=400, detail="Missing action_type")
    try:
        action_type = ActionType(action_type_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported action_type {action_type_raw}") from exc
    observation, reward, done, info = env.step(Action(action_type=action_type, payload=request.resolved_payload()))
    return OpenEnvResponseEnvelope(
        session_id=request.session_id,
        observation=_serialize_observation(observation),
        reward=float(reward),
        done=bool(done),
        info=_safe_session_info(info),
    )


@app.get("/openenv/state/{session_id}", response_model=OpenEnvResponseEnvelope)
def openenv_state(session_id: str) -> OpenEnvResponseEnvelope:
    env = _get_session_env(session_id)
    if env.state is None:
        raise HTTPException(status_code=400, detail="Session has not been reset yet")
    return OpenEnvResponseEnvelope(
        session_id=session_id,
        observation=_serialize_observation(env._observation()),
        reward=0.0,
        done=bool(env.state.done),
        info=_safe_session_info(env._info()),
    )


@app.post("/openenv/report_inference", response_model=OpenEnvInferenceReportResponse)
def openenv_report_inference(request: OpenEnvInferenceReportRequest) -> OpenEnvInferenceReportResponse:
    env = _build_environment()
    normalized_episodes = _normalize_episode_rows(env, list(request.episodes))
    payload = {
        "run": dict(request.run),
        "summary": dict(request.summary),
        "episodes": normalized_episodes,
    }
    LATEST_EVALUATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LATEST_EVALUATION_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    dashboard_path = export_dashboard(
        env=env,
        evaluation=payload,
        leaderboard_records=load_leaderboard("artifacts/baselines/openai_fixed_levels_leaderboard.json"),
        output_path=str(SPACE_DASHBOARD),
    )
    return OpenEnvInferenceReportResponse(
        status="ok",
        output_path=str(LATEST_EVALUATION_OUTPUT),
        dashboard_path=str(dashboard_path),
    )


@app.get("/dashboard")
def dashboard() -> FileResponse:
    snapshot = _space_snapshot()
    return FileResponse(snapshot["dashboard_path"], media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=SPACE_PORT)
