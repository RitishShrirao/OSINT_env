#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from osint_env.config import clone_environment_config, load_shared_config
from osint_env.env.environment import OSINTEnvironment
from osint_env.llm import build_llm_client
from osint_env.training import load_self_play_config
from osint_env.training.self_play import _run_post_training_evaluation
from osint_env.viz import export_dashboard


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest self-play checkpoint from Hugging Face, "
            "generate fresh questions, compare the finetuned checkpoint "
            "against the base model, and export benchmark-style HTML."
        )
    )
    parser.add_argument("--repo-id", required=True, help="HF repo id, for example Siddeshwar1625/osint-checkpoints.")
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="Run folder inside the HF repo, for example self_play_hf_l40s_full.",
    )
    parser.add_argument("--repo-type", default="model", help="HF repo type. Defaults to model.")
    parser.add_argument("--env-config", default="config/shared_config.json", help="Shared environment config.")
    parser.add_argument(
        "--train-config",
        default="config/self_play_training_hf_l40s_full.json",
        help="Self-play training config used for question generation and compare settings.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/hf_checkpoint_eval",
        help="Directory where evaluation JSON and HTML artifacts will be written.",
    )
    parser.add_argument(
        "--download-dir",
        default="artifacts/hf_downloads",
        help="Directory used for local HF downloads and cache materialization.",
    )
    parser.add_argument(
        "--dashboard-name",
        default="post_training_benchmark_dashboard.html",
        help="Filename for the finetuned benchmark-style HTML dashboard.",
    )
    parser.add_argument(
        "--original-dashboard-name",
        default="post_training_benchmark_dashboard_original.html",
        help="Filename for the base-model benchmark-style HTML dashboard.",
    )
    parser.add_argument(
        "--leaderboard-name",
        default="post_training_compare_leaderboard.json",
        help="Filename for the two-row leaderboard JSON used by the HTML dashboards.",
    )
    parser.add_argument(
        "--base-model",
        default="",
        help="Optional base model override. Defaults to the model recorded in self_play_summary.json.",
    )
    parser.add_argument(
        "--finetuned-model-subpath",
        default="",
        help=(
            "Optional HF path to the finetuned model directory inside the repo. "
            "Defaults to the final answerer model recorded in self_play_summary.json."
        ),
    )
    parser.add_argument(
        "--env-llm-provider",
        default="mock",
        help="Provider used only for environment construction. Defaults to mock.",
    )
    parser.add_argument(
        "--allow-env-llm-seeding",
        action="store_true",
        help=(
            "Keep graph/task LLM seeding enabled while constructing the environment. "
            "By default this script disables it to avoid depending on a local LLM server."
        ),
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=0,
        help="Optional override for post_training_eval_questions.",
    )
    parser.add_argument(
        "--generated-task-max-new-tokens",
        type=int,
        default=0,
        help="Optional override for generated_task_max_new_tokens.",
    )
    parser.add_argument(
        "--answer-max-new-tokens",
        type=int,
        default=0,
        help="Optional override for post_training_eval_answer_max_new_tokens.",
    )
    return parser


def _strip_artifacts_prefix(path_value: str) -> str:
    path = Path(str(path_value).strip())
    parts = path.parts
    if parts and parts[0] == "artifacts":
        return Path(*parts[1:]).as_posix()
    return path.as_posix()


def _resolve_finetuned_model_subpath(summary: dict[str, Any], explicit: str) -> str:
    if explicit.strip():
        return explicit.strip().strip("/")

    final_models = summary.get("final_models", {}) if isinstance(summary, dict) else {}
    candidate = str(final_models.get("answerer") or final_models.get("generator") or "").strip()
    if not candidate:
        raise ValueError("Could not resolve final model path from self_play_summary.json.")
    return _strip_artifacts_prefix(candidate)


def _load_summary(repo_id: str, repo_type: str, run_prefix: str, download_dir: Path) -> tuple[Path, dict[str, Any]]:
    local_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=f"{run_prefix.strip('/')}/self_play_summary.json",
            local_dir=str(download_dir),
        )
    )
    payload = json.loads(local_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("self_play_summary.json did not contain a JSON object.")
    return local_path, payload


def _download_model_dir(repo_id: str, repo_type: str, model_subpath: str, download_dir: Path) -> Path:
    normalized = model_subpath.strip().strip("/")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=[f"{normalized}/*"],
        local_dir=str(download_dir),
    )
    local_model_dir = download_dir / normalized
    if not local_model_dir.exists():
        raise FileNotFoundError(f"Downloaded model folder not found: {local_model_dir}")
    return local_model_dir


def _benchmark_like_summary(summary: dict[str, Any]) -> dict[str, float]:
    task_success_rate = float(summary.get("task_success_rate", 0.0))
    avg_graph_f1 = float(summary.get("avg_graph_f1", 0.0))
    avg_reward = float(summary.get("avg_reward", 0.0))
    leaderboard_score = (
        0.28 * task_success_rate
        + 0.20 * avg_graph_f1
        + 0.05 * avg_reward
    )
    return {
        "task_success_rate": task_success_rate,
        "tool_efficiency": 0.0,
        "avg_graph_f1": avg_graph_f1,
        "avg_steps_to_solution": 0.0,
        "deanonymization_accuracy": 0.0,
        "avg_reward": avg_reward,
        "avg_knowledge_carrier_reward": 0.0,
        "avg_knowledge_indexing_reward": 0.0,
        "avg_connectivity_reward": 0.0,
        "avg_format_reward": 0.0,
        "avg_relation_informativeness_reward": 0.0,
        "avg_entity_informativeness_reward": 0.0,
        "avg_diversity_reward": 0.0,
        "avg_soft_shaping_reward": 0.0,
        "avg_connectivity_gain_reward": 0.0,
        "avg_compactness_reward": 0.0,
        "avg_spawn_count": 0.0,
        "spawn_completion_rate": 0.0,
        "avg_spawn_critical_steps": 0.0,
        "spawn_signal": 0.0,
        "retrieval_signal": 0.0,
        "structural_signal": 0.0,
        "leaderboard_score": leaderboard_score,
    }


def _benchmark_like_evaluation(
    payload: dict[str, Any],
    model_label: str,
) -> dict[str, Any]:
    model_evaluations = payload.get("model_evaluations", {}) if isinstance(payload, dict) else {}
    model_payload = model_evaluations.get(model_label, {}) if isinstance(model_evaluations, dict) else {}
    summary = model_payload.get("summary", {}) if isinstance(model_payload, dict) else {}
    episodes = model_payload.get("episodes", []) if isinstance(model_payload, dict) else []

    benchmark_episodes: list[dict[str, Any]] = []
    for episode in episodes if isinstance(episodes, list) else []:
        if not isinstance(episode, dict):
            continue
        benchmark_episodes.append(
            {
                "task_id": str(episode.get("task_id", "")),
                "task_type": str(episode.get("task_type", "")),
                "question": str(episode.get("question", "")),
                "task_answer": str(episode.get("task_answer", "")),
                "agent_answer": str(episode.get("agent_answer", "")),
                "graph_f1": float(episode.get("graph_f1", 0.0)),
                "reward": float(episode.get("reward", 0.0)),
                "steps": 0,
                "tool_calls": 0,
                "success": int(episode.get("success", 0)),
                "reward_components": {},
                "spawn_count": 0,
                "spawn_critical_steps": 0,
                "pred_edges": list(episode.get("pred_edges", [])),
                "truth_edges": list(episode.get("truth_edges", [])),
            }
        )

    return {
        "summary": _benchmark_like_summary(summary if isinstance(summary, dict) else {}),
        "episodes": benchmark_episodes,
    }


def _leaderboard_records(compare_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, model_label in enumerate(("finetuned_answerer", "original_answerer"), start=1):
        evaluation = _benchmark_like_evaluation(compare_payload, model_label)
        records.append(
            {
                "run_id": f"post_train_{idx:02d}",
                "run_name": model_label,
                "episodes": len(evaluation.get("episodes", [])),
                "config": {"source": "post_training_evaluation"},
                "metrics": evaluation.get("summary", {}),
            }
        )
    return records


def main() -> None:
    args = _build_parser().parse_args()

    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path, summary = _load_summary(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        run_prefix=args.run_prefix,
        download_dir=download_dir,
    )
    finetuned_model_subpath = _resolve_finetuned_model_subpath(summary, args.finetuned_model_subpath)
    finetuned_model_dir = _download_model_dir(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        model_subpath=finetuned_model_subpath,
        download_dir=download_dir,
    )

    train_cfg = load_self_play_config(args.train_config)
    if args.questions > 0:
        train_cfg.post_training_eval_questions = int(args.questions)
    if args.generated_task_max_new_tokens > 0:
        train_cfg.generated_task_max_new_tokens = int(args.generated_task_max_new_tokens)
    if args.answer_max_new_tokens > 0:
        train_cfg.post_training_eval_answer_max_new_tokens = int(args.answer_max_new_tokens)

    shared_cfg = load_shared_config(args.env_config)
    env_cfg = clone_environment_config(shared_cfg.environment)
    env_cfg.llm.provider = str(args.env_llm_provider).strip() or "mock"
    if not args.allow_env_llm_seeding:
        env_cfg.seeding.llm_generate_remaining_graph = False
        env_cfg.seeding.llm_generate_remaining_tasks = False

    base_model = str(args.base_model).strip() or str(
        summary.get("initial_models", {}).get("answerer")
        or summary.get("initial_models", {}).get("generator")
        or train_cfg.shared_model_name_or_path
    )
    pipeline_mode = str(summary.get("pipeline_mode") or train_cfg.pipeline_mode or "swarm_v2")

    compare_payload = _run_post_training_evaluation(
        env_config=env_cfg,
        training_config=train_cfg,
        generator_model=str(finetuned_model_dir),
        answerer_models={
            "finetuned_answerer": str(finetuned_model_dir),
            "original_answerer": base_model,
        },
        output_dir=output_dir,
        pipeline_mode=pipeline_mode,
        effective_dry_run=False,
    )

    env = OSINTEnvironment(env_cfg, llm=build_llm_client(env_cfg.llm))
    env.reset()

    leaderboard_records = _leaderboard_records(compare_payload)
    leaderboard_path = output_dir / args.leaderboard_name
    leaderboard_path.write_text(json.dumps(leaderboard_records, indent=2, sort_keys=True), encoding="utf-8")

    finetuned_eval = _benchmark_like_evaluation(compare_payload, "finetuned_answerer")
    original_eval = _benchmark_like_evaluation(compare_payload, "original_answerer")

    finetuned_dashboard_path = output_dir / args.dashboard_name
    original_dashboard_path = output_dir / args.original_dashboard_name
    export_dashboard(env=env, evaluation=finetuned_eval, leaderboard_records=leaderboard_records, output_path=str(finetuned_dashboard_path))
    export_dashboard(env=env, evaluation=original_eval, leaderboard_records=leaderboard_records, output_path=str(original_dashboard_path))

    context = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "run_prefix": args.run_prefix,
        "summary_path": str(summary_path),
        "downloaded_finetuned_model": str(finetuned_model_dir),
        "base_model": base_model,
        "pipeline_mode": pipeline_mode,
        "environment_llm_provider": env_cfg.llm.provider,
        "env_llm_seeding_enabled": bool(args.allow_env_llm_seeding),
        "dashboard_paths": {
            "finetuned": str(finetuned_dashboard_path),
            "original": str(original_dashboard_path),
        },
        "leaderboard_path": str(leaderboard_path),
        "evaluation_path": str(compare_payload.get("path", "")),
    }
    (output_dir / "evaluation_context.json").write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "evaluation_path": compare_payload.get("path", ""),
                "dashboard_path": str(finetuned_dashboard_path),
                "original_dashboard_path": str(original_dashboard_path),
                "leaderboard_path": str(leaderboard_path),
                "summary": compare_payload.get("summary", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
