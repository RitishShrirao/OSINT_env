from pathlib import Path

from osint_env.eval.leaderboard import append_leaderboard_record, load_leaderboard, render_leaderboard_table, sorted_leaderboard


def test_leaderboard_roundtrip(tmp_path: Path):
    board = tmp_path / "leaderboard.json"
    append_leaderboard_record(
        path=board,
        summary={
            "leaderboard_score": 0.42,
            "task_success_rate": 0.5,
            "avg_graph_f1": 0.4,
            "avg_reward": 0.1,
            "tool_efficiency": 0.9,
            "retrieval_signal": 0.3,
            "structural_signal": 0.4,
        },
        episodes=5,
        run_name="baseline",
    )
    append_leaderboard_record(
        path=board,
        summary={
            "leaderboard_score": 0.75,
            "task_success_rate": 0.7,
            "avg_graph_f1": 0.6,
            "avg_reward": 0.5,
            "tool_efficiency": 0.8,
            "retrieval_signal": 0.6,
            "structural_signal": 0.7,
        },
        episodes=5,
        run_name="improved",
    )

    records = load_leaderboard(board)
    ranked = sorted_leaderboard(records)
    assert len(records) == 2
    assert ranked[0]["run_name"] == "improved"

    ranked_by_success = sorted_leaderboard(records, sort_by="task_success_rate")
    assert ranked_by_success[0]["run_name"] == "improved"

    table = render_leaderboard_table(records, top_k=5)
    assert "| rank | run |" in table
    assert "retrieval" in table
