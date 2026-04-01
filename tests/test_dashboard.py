from pathlib import Path

from osint_env.domain.models import EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment
from osint_env.viz import export_dashboard


def test_dashboard_export(tmp_path: Path):
    env = OSINTEnvironment(EnvironmentConfig(seed=9, n_users=14))
    env.reset()

    out = tmp_path / "dashboard.html"
    path = export_dashboard(
        env=env,
        evaluation={"summary": {"leaderboard_score": 0.0, "task_success_rate": 0.0, "avg_graph_f1": 0.0, "tool_efficiency": 0.0, "deanonymization_accuracy": 0.0, "avg_reward": 0.0}, "episodes": []},
        leaderboard_records=[],
        output_path=str(out),
    )

    assert path.endswith("dashboard.html")
    text = out.read_text(encoding="utf-8")
    assert "OSINT Benchmark Dashboard" in text
    assert "Canonical Graph" in text
    assert "Original Database Explorer" in text
    assert "Benchmark Leaderboard" in text
