from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def load_leaderboard(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def save_leaderboard(path: str | Path, records: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def _metric_value(record: dict[str, Any], sort_by: str) -> float:
    metrics = record.get("metrics", {})
    return float(metrics.get(sort_by, 0.0))


def sorted_leaderboard(records: list[dict[str, Any]], sort_by: str = "leaderboard_score") -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: _metric_value(r, sort_by), reverse=True)


def append_leaderboard_record(
    path: str | Path,
    summary: dict[str, Any],
    episodes: int,
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = load_leaderboard(path)
    run_id = f"run_{len(records) + 1:04d}"
    record = {
        "run_id": run_id,
        "run_name": run_name or run_id,
        "created_at": _utc_now(),
        "episodes": int(episodes),
        "config": config or {},
        "metrics": summary,
    }
    records.append(record)
    save_leaderboard(path, records)
    return record


def render_leaderboard_table(records: list[dict[str, Any]], top_k: int = 20, sort_by: str = "leaderboard_score") -> str:
    ranked = sorted_leaderboard(records, sort_by=sort_by)[:top_k]
    header = "| rank | run | score | success | graph_f1 | retrieval | structural | spawn | reward | tool_eff |\n"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    rows: list[str] = []
    for idx, rec in enumerate(ranked, start=1):
        m = rec.get("metrics", {})
        rows.append(
            "| {rank} | {run} | {score:.4f} | {succ:.3f} | {f1:.3f} | {retrieval:.3f} | {structural:.3f} | {spawn:.3f} | {reward:.3f} | {tool:.3f} |".format(
                rank=idx,
                run=rec.get("run_name", rec.get("run_id", "run")),
                score=float(m.get("leaderboard_score", 0.0)),
                succ=float(m.get("task_success_rate", 0.0)),
                f1=float(m.get("avg_graph_f1", 0.0)),
                retrieval=float(m.get("retrieval_signal", 0.0)),
                structural=float(m.get("structural_signal", 0.0)),
                spawn=float(m.get("spawn_signal", 0.0)),
                reward=float(m.get("avg_reward", 0.0)),
                tool=float(m.get("tool_efficiency", 0.0)),
            )
        )
    return header + sep + "\n".join(rows)
