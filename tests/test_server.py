import json
import os

from fastapi.testclient import TestClient

import server
from server import app


client = TestClient(app)


def test_server_health():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_server_environment_metadata():
    response = client.get("/api/environment")
    assert response.status_code == 200
    body = response.json()
    assert "action_space" in body
    assert "observation_space" in body
    assert "summary" in body


def test_openenv_spec_and_tasks_endpoints():
    spec = client.get("/openenv.yaml")
    assert spec.status_code == 200
    assert "reset" in spec.text

    tasks = client.get("/openenv/tasks")
    assert tasks.status_code == 200
    body = tasks.json()
    assert len(body) >= 3
    assert {"task_id", "task_type", "question", "difficulty"} <= set(body[0].keys())


def test_openenv_reset_step_and_state_cycle():
    reset = client.post("/openenv/reset", json={"task_index": 0})
    assert reset.status_code == 200
    body = reset.json()
    session_id = body["session_id"]
    assert body["done"] is False
    assert "question" in body["observation"]["task"]

    state = client.get(f"/openenv/state/{session_id}")
    assert state.status_code == 200
    assert state.json()["session_id"] == session_id

    step = client.post(
        "/openenv/step",
        json={
            "session_id": session_id,
            "action_type": "ANSWER",
            "payload": {"answer": "unknown"},
        },
    )
    assert step.status_code == 200
    step_body = step.json()
    assert step_body["session_id"] == session_id
    assert step_body["done"] is True
    assert "task_answer" in step_body["info"]


def test_openenv_reset_accepts_empty_body():
    reset = client.post("/openenv/reset")
    assert reset.status_code == 200
    body = reset.json()
    assert body["done"] is False
    assert "session_id" in body


def test_openenv_reset_accepts_empty_json_body():
    reset = client.post(
        "/openenv/reset",
        data="",
        headers={"Content-Type": "application/json"},
    )
    assert reset.status_code == 200
    body = reset.json()
    assert body["done"] is False
    assert "session_id" in body


def test_openenv_reset_trailing_slash_post_returns_json():
    reset = client.post(
        "/openenv/reset/",
        data="",
        headers={"Content-Type": "application/json"},
    )
    assert reset.status_code == 200
    body = reset.json()
    assert body["done"] is False
    assert "session_id" in body


def test_openenv_step_accepts_nested_action_payload():
    reset = client.post("/openenv/reset", json={"task_index": 0})
    assert reset.status_code == 200
    session_id = reset.json()["session_id"]

    step = client.post(
        "/openenv/step",
        json={
            "session_id": session_id,
            "action": {
                "action_type": "ANSWER",
                "payload": {"answer": "unknown"},
            },
        },
    )
    assert step.status_code == 200
    assert step.json()["done"] is True


def test_report_inference_updates_latest_evaluation_and_dashboard(tmp_path, monkeypatch):
    latest_evaluation = tmp_path / "latest_evaluation.json"
    space_dashboard = tmp_path / "space_dashboard.html"

    monkeypatch.setattr(server, "LATEST_EVALUATION_OUTPUT", latest_evaluation)
    monkeypatch.setattr(server, "SPACE_DASHBOARD", space_dashboard)
    monkeypatch.setattr(server, "load_leaderboard", lambda path: [])
    monkeypatch.setattr(server, "export_dashboard", lambda env, evaluation, leaderboard_records, output_path: str(space_dashboard))

    response = client.post(
        "/openenv/report_inference",
        json={
            "run": {"name": "inference_py_run"},
            "summary": {"leaderboard_score": 0.75, "task_success_rate": 1.0},
            "episodes": [
                {
                    "task_id": "seed_task_0",
                    "agent_answer": "user_bharat",
                    "graph_f1": 0.5,
                    "reward": 1.2,
                    "steps": 5,
                    "tool_calls": 4,
                    "success": 1,
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert latest_evaluation.exists()
    stored = json.loads(latest_evaluation.read_text(encoding="utf-8"))
    assert stored["summary"]["leaderboard_score"] == 0.75
    assert stored["episodes"][0]["task_id"] == "seed_task_0"
    assert stored["episodes"][0]["truth_edges"]


def test_space_snapshot_prefers_newer_evaluation_payload(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.json"
    evaluation_path = tmp_path / "evaluation.json"
    baseline_dashboard = tmp_path / "baseline_dashboard.html"
    space_dashboard = tmp_path / "space_dashboard.html"

    baseline_path.write_text(
        json.dumps(
            {
                "run": {"dashboard_path": str(baseline_dashboard)},
                "summary": {"leaderboard_score": 0.1, "task_success_rate": 0.1},
            }
        ),
        encoding="utf-8",
    )
    baseline_dashboard.write_text("<html>baseline</html>", encoding="utf-8")
    evaluation_path.write_text(
        json.dumps({"summary": {"leaderboard_score": 0.9, "task_success_rate": 0.9}, "episodes": []}),
        encoding="utf-8",
    )
    space_dashboard.write_text("<html>space</html>", encoding="utf-8")
    os.utime(evaluation_path, (baseline_path.stat().st_atime + 5, baseline_path.stat().st_mtime + 5))

    monkeypatch.setattr(server, "LATEST_BASELINE_OUTPUT", baseline_path)
    monkeypatch.setattr(server, "LATEST_EVALUATION_OUTPUT", evaluation_path)
    monkeypatch.setattr(server, "SPACE_DASHBOARD", space_dashboard)
    monkeypatch.setattr(
        server,
        "_base_environment_snapshot",
        lambda: {
            "task_count": 30,
            "difficulty_counts": {},
            "action_space": ["CALL_TOOL", "ADD_EDGE", "ANSWER"],
            "observation_space": {},
            "task_types": [],
            "config": {},
        },
    )
    monkeypatch.setattr(server, "_build_environment", lambda: object())
    monkeypatch.setattr(server, "export_dashboard", lambda env, evaluation, leaderboard_records, output_path: str(space_dashboard))

    snapshot = server._space_snapshot()
    assert snapshot["source"] == "latest_evaluation"
    assert snapshot["summary"]["leaderboard_score"] == 0.9
    assert snapshot["dashboard_path"] == str(space_dashboard)
