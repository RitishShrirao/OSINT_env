from __future__ import annotations

import argparse

from osint_env import cli
from osint_env.domain.models import EnvironmentConfig


class _DummyParser:
    def __init__(self, namespace: argparse.Namespace):
        self._namespace = namespace

    def parse_args(self) -> argparse.Namespace:
        return self._namespace


class _DummyEnv:
    def __init__(self, config: EnvironmentConfig, llm=None):
        self.config = config
        self.llm = llm


def test_eval_exports_dashboard_and_evaluation(monkeypatch, tmp_path, capsys):
    dashboard_path = tmp_path / "eval_dashboard.html"
    eval_path = tmp_path / "latest_evaluation.json"

    args = argparse.Namespace(
        cmd="eval",
        episodes=1,
        leaderboard="",
        dashboard="",
        dashboard_dir="",
        evaluation="",
    )

    runtime = {
        "default_episodes": 20,
        "leaderboard_path": str(tmp_path / "leaderboard.json"),
        "dashboard_path": str(dashboard_path),
        "sweep_dashboard_dir": str(tmp_path / "sweep"),
    }

    evaluation_payload = {
        "summary": {
            "avg_reward": 0.5,
            "avg_graph_f1": 0.4,
            "task_success_rate": 1.0,
            "tool_efficiency": 0.7,
            "avg_steps_to_solution": 3.0,
            "deanonymization_accuracy": 1.0,
            "leaderboard_score": 0.8,
        },
        "episodes": [
            {
                "task_id": "metaqa_1-hop_train_0",
                "task_type": "metaqa_1-hop",
                "question": "who directed [inception]?",
                "task_answer": "christopher nolan",
                "agent_answer": "christopher nolan",
                "graph_f1": 1.0,
                "reward": 1.0,
                "steps": 2,
                "tool_calls": 1,
                "success": 1,
            }
        ],
    }

    calls: dict[str, object] = {}

    monkeypatch.setattr(cli, "build_parser", lambda: _DummyParser(args))
    monkeypatch.setattr(cli, "_resolve_environment_config", lambda _args: (EnvironmentConfig(), runtime))
    monkeypatch.setattr(cli, "build_llm_client", lambda _cfg: object())
    monkeypatch.setattr(cli, "OSINTEnvironment", _DummyEnv)
    monkeypatch.setattr(cli, "run_evaluation", lambda env, episodes, return_details, llm: evaluation_payload)

    def _save(path: str, payload: dict) -> None:
        calls["save_path"] = path
        calls["save_payload"] = payload

    def _export(env, evaluation, leaderboard_records, output_path):
        calls["export_output_path"] = output_path
        calls["export_eval"] = evaluation
        calls["export_leaderboard"] = leaderboard_records
        return output_path

    monkeypatch.setattr(cli, "_save_evaluation", _save)
    monkeypatch.setattr(cli, "load_leaderboard", lambda _path: [])
    monkeypatch.setattr(cli, "export_dashboard", _export)

    monkeypatch.setattr(cli, "DEFAULT_EVALUATION_PATH", str(eval_path))

    cli.main()

    assert calls["save_path"] == str(eval_path)
    assert calls["save_payload"] == evaluation_payload
    assert calls["export_output_path"] == str(dashboard_path)
    assert calls["export_eval"] == evaluation_payload
    assert calls["export_leaderboard"] == []

    output = capsys.readouterr().out
    assert '"avg_reward": 0.5' in output
    assert '"episodes"' not in output
