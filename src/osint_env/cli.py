from __future__ import annotations

import argparse
import json
from pathlib import Path

from osint_env.agents.single_agent import SingleAgentRunner
from osint_env.agents.swarm_agent import SwarmAgentRunner
from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.domain.models import EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.reward import compute_graph_f1
from osint_env.eval.leaderboard import append_leaderboard_record, load_leaderboard, render_leaderboard_table
from osint_env.eval.runner import run_evaluation
from osint_env.llm import build_llm_client
from osint_env.viz import export_dashboard


DEFAULT_EVALUATION_PATH = "artifacts/latest_evaluation.json"


def _save_evaluation(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_evaluation(path: str) -> dict | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="config/shared_config.json")
    parser.add_argument("--seed-file", type=str, default="")
    parser.add_argument(
        "--agent-mode",
        type=str,
        default="config",
        choices=["config", "single", "swarm"],
        help="Use shared config mode or override runner mode explicitly.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="config",
        choices=["config", "mock", "ollama", "openai"],
        help="Use shared config provider or override explicitly.",
    )
    parser.add_argument("--llm-model", type=str, default="", help="Override model name for selected LLM provider.")
    parser.add_argument("--llm-timeout-seconds", type=int, default=0, help="Override LLM request timeout in seconds.")
    parser.add_argument("--ollama-base-url", type=str, default="", help="Override Ollama base URL.")
    parser.add_argument("--openai-base-url", type=str, default="", help="Override OpenAI base URL.")
    parser.add_argument("--openai-api-key", type=str, default="", help="OpenAI API key override.")
    parser.add_argument(
        "--openai-api-key-env",
        type=str,
        default="",
        help="Environment variable name for OpenAI API key.",
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="config",
        choices=["config", "canonical", "metaqa"],
        help="Use dataset mode from config or override with canonical/metaqa.",
    )
    parser.add_argument("--metaqa-root", type=str, default="", help="Override MetaQA dataset root directory.")
    parser.add_argument(
        "--metaqa-kb-path",
        type=str,
        default="",
        help="Override MetaQA KB triples file path. Defaults to <metaqa-root>/kb.txt.",
    )
    parser.add_argument(
        "--metaqa-variant",
        type=str,
        default="",
        choices=["", "vanilla", "ntm"],
        help="Override MetaQA QA variant.",
    )
    parser.add_argument(
        "--metaqa-hops",
        type=str,
        default="",
        help="Comma-separated hop buckets for MetaQA mode (example: 1-hop,2-hop,3-hop).",
    )
    parser.add_argument(
        "--metaqa-splits",
        type=str,
        default="",
        help="Comma-separated splits for MetaQA mode (example: train,dev,test).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osint-env")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Run one episode and print debug info.")
    _add_common_args(d)

    e = sub.add_parser("eval", help="Run multiple episodes and show aggregate metrics.")
    _add_common_args(e)
    e.add_argument("--episodes", type=int, default=0)
    e.add_argument("--dashboard", type=str, default="")

    b = sub.add_parser("benchmark", help="Run eval, update leaderboard, and export interactive dashboard.")
    _add_common_args(b)
    b.add_argument("--episodes", type=int, default=0)
    b.add_argument("--name", type=str, default="")
    b.add_argument("--leaderboard", type=str, default="")
    b.add_argument("--dashboard", type=str, default="")

    l = sub.add_parser("leaderboard", help="Print ranked benchmark leaderboard.")
    _add_common_args(l)
    l.add_argument("--leaderboard", type=str, default="")
    l.add_argument("--top", type=int, default=20)
    l.add_argument(
        "--sort-by",
        type=str,
        default="leaderboard_score",
        choices=[
            "leaderboard_score",
            "task_success_rate",
            "avg_graph_f1",
            "tool_efficiency",
            "avg_reward",
            "retrieval_signal",
            "structural_signal",
            "deanonymization_accuracy",
            "spawn_signal",
        ],
    )

    s = sub.add_parser("benchmark-sweep", help="Run benchmark across multiple seeds and append all runs to leaderboard.")
    _add_common_args(s)
    s.add_argument("--episodes", type=int, default=0)
    s.add_argument("--seeds", type=str, default="7,11,17,23,31")
    s.add_argument("--name-prefix", type=str, default="sweep")
    s.add_argument("--leaderboard", type=str, default="")
    s.add_argument("--dashboard-dir", type=str, default="")

    v = sub.add_parser("viz", help="Export an interactive graph/database explorer.")
    _add_common_args(v)
    v.add_argument("--output", type=str, default="artifacts/osint_explorer.html")
    v.add_argument("--with-demo", action="store_true")
    v.add_argument("--leaderboard", type=str, default="")
    v.add_argument(
        "--evaluation",
        type=str,
        default=DEFAULT_EVALUATION_PATH,
        help="Path to a saved evaluation payload with episode details.",
    )
    return parser


def _resolve_environment_config(args: argparse.Namespace) -> tuple[EnvironmentConfig, dict[str, str | int]]:
    shared = load_shared_config(args.config)
    env_cfg = clone_environment_config(shared.environment)

    if args.seed_file:
        env_cfg.seeding = load_seeding_config(args.seed_file)

    if args.llm_provider != "config":
        env_cfg.llm.provider = args.llm_provider
    if args.llm_model:
        env_cfg.llm.model = args.llm_model
    if int(args.llm_timeout_seconds) > 0:
        env_cfg.llm.timeout_seconds = int(args.llm_timeout_seconds)
    if args.ollama_base_url:
        env_cfg.llm.ollama_base_url = args.ollama_base_url
    if args.openai_base_url:
        env_cfg.llm.openai_base_url = args.openai_base_url
    if args.openai_api_key:
        env_cfg.llm.openai_api_key = args.openai_api_key
    if args.openai_api_key_env:
        env_cfg.llm.openai_api_key_env = args.openai_api_key_env

    if args.dataset_mode != "config":
        env_cfg.dataset_mode = args.dataset_mode
    if args.metaqa_root:
        env_cfg.metaqa_root = args.metaqa_root
    if args.metaqa_kb_path:
        env_cfg.metaqa_kb_path = args.metaqa_kb_path
    if args.metaqa_variant:
        env_cfg.metaqa_variant = args.metaqa_variant
    if args.metaqa_hops:
        env_cfg.metaqa_hops = [item.strip() for item in str(args.metaqa_hops).split(",") if item.strip()]
    if args.metaqa_splits:
        env_cfg.metaqa_splits = [item.strip() for item in str(args.metaqa_splits).split(",") if item.strip()]

    if args.agent_mode == "single":
        env_cfg.swarm.enabled = False
    elif args.agent_mode == "swarm":
        env_cfg.swarm.enabled = True

    runtime = {
        "default_episodes": shared.runtime.default_episodes,
        "leaderboard_path": shared.runtime.leaderboard_path,
        "dashboard_path": shared.runtime.dashboard_path,
        "sweep_dashboard_dir": shared.runtime.sweep_dashboard_dir,
    }
    return env_cfg, runtime


def _runner_for(env: OSINTEnvironment) -> SingleAgentRunner | SwarmAgentRunner:
    if env.config.swarm.enabled:
        return SwarmAgentRunner(env, llm=build_llm_client(env.config.llm))
    return SingleAgentRunner(env, llm=build_llm_client(env.config.llm))


def main() -> None:
    args = build_parser().parse_args()
    env_cfg, runtime = _resolve_environment_config(args)

    episodes = int(args.episodes) if getattr(args, "episodes", 0) else int(runtime["default_episodes"])
    leaderboard_path = str(args.leaderboard) if getattr(args, "leaderboard", "") else str(runtime["leaderboard_path"])
    dashboard_path = str(args.dashboard) if getattr(args, "dashboard", "") else str(runtime["dashboard_path"])
    sweep_dashboard_dir = (
        str(args.dashboard_dir) if getattr(args, "dashboard_dir", "") else str(runtime["sweep_dashboard_dir"])
    )
    evaluation_path = str(getattr(args, "evaluation", "") or DEFAULT_EVALUATION_PATH)

    if args.cmd == "leaderboard":
        records = load_leaderboard(leaderboard_path)
        print(render_leaderboard_table(records, top_k=args.top, sort_by=args.sort_by))
        return

    if args.cmd == "benchmark-sweep":
        seed_values = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        outputs: list[dict[str, object]] = []
        for seed in seed_values:
            seeded_cfg = clone_environment_config(env_cfg)
            seeded_cfg.seed = seed
            env = OSINTEnvironment(seeded_cfg, llm=build_llm_client(seeded_cfg.llm))
            evaluation = run_evaluation(env, episodes=episodes, return_details=True, llm=build_llm_client(seeded_cfg.llm))
            summary = evaluation["summary"]
            run_name = f"{args.name_prefix}_seed{seed}"
            record = append_leaderboard_record(
                path=leaderboard_path,
                summary=summary,
                episodes=episodes,
                run_name=run_name,
                config={
                    "seed": seed,
                    "max_steps": env.config.max_steps,
                    "swarm_enabled": env.config.swarm.enabled,
                    "max_agents": env.config.swarm.max_agents,
                    "max_breadth": env.config.swarm.max_breadth,
                    "max_width": env.config.swarm.max_width,
                    "max_depth": env.config.swarm.max_depth,
                    "seeded_questions": len(env.config.seeding.seeded_questions),
                },
            )
            dashboard_path = export_dashboard(
                env=env,
                evaluation=evaluation,
                leaderboard_records=load_leaderboard(leaderboard_path),
                output_path=f"{sweep_dashboard_dir}/{run_name}.html",
            )
            _save_evaluation(DEFAULT_EVALUATION_PATH, evaluation)
            outputs.append({"seed": seed, "record": record, "dashboard": dashboard_path, "summary": summary})

        records = load_leaderboard(leaderboard_path)
        print(
            json.dumps(
                {
                    "runs": outputs,
                    "leaderboard_preview": render_leaderboard_table(records, top_k=min(10, len(records))),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    llm_client = build_llm_client(env_cfg.llm)
    env = OSINTEnvironment(env_cfg, llm=llm_client)
    if args.cmd == "demo":
        info = _runner_for(env).run_episode()
        print(json.dumps(info, indent=2, sort_keys=True))
    elif args.cmd == "eval":
        evaluation = run_evaluation(env, episodes=episodes, return_details=True, llm=llm_client)
        _save_evaluation(DEFAULT_EVALUATION_PATH, evaluation)
        leaderboard = load_leaderboard(leaderboard_path)
        export_dashboard(
            env=env,
            evaluation=evaluation,
            leaderboard_records=leaderboard,
            output_path=dashboard_path,
        )
        print(json.dumps(evaluation["summary"], indent=2, sort_keys=True))
    elif args.cmd == "benchmark":
        evaluation = run_evaluation(env, episodes=episodes, return_details=True, llm=llm_client)
        summary = evaluation["summary"]
        record = append_leaderboard_record(
            path=leaderboard_path,
            summary=summary,
            episodes=episodes,
            run_name=args.name or None,
            config={
                "seed": env.config.seed,
                "max_steps": env.config.max_steps,
                "swarm_enabled": env.config.swarm.enabled,
                "max_agents": env.config.swarm.max_agents,
                "max_breadth": env.config.swarm.max_breadth,
                "max_width": env.config.swarm.max_width,
                "max_depth": env.config.swarm.max_depth,
                "seeded_questions": len(env.config.seeding.seeded_questions),
            },
        )
        leaderboard = load_leaderboard(leaderboard_path)
        dashboard_path = export_dashboard(
            env=env,
            evaluation=evaluation,
            leaderboard_records=leaderboard,
            output_path=dashboard_path,
        )
        _save_evaluation(DEFAULT_EVALUATION_PATH, evaluation)
        payload = {
            "record": record,
            "summary": summary,
            "dashboard": dashboard_path,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.cmd == "viz":
        evaluation: dict | None = _load_evaluation(evaluation_path)
        if args.with_demo:
            _runner_for(env).run_episode()
            info = {
                "agent_answer": env.state.answer if env.state else "",
                "task_answer": env.state.task.answer if env.state else "",
                "total_reward": env.state.total_reward if env.state else 0.0,
                "step_count": env.state.step_count if env.state else 0,
                "tool_calls": env.state.tool_calls if env.state else 0,
            }
            evaluation = {
                "summary": {
                    "task_success_rate": float(info["agent_answer"] == info["task_answer"]),
                    "tool_efficiency": 0.0,
                    "avg_graph_f1": 0.0,
                    "avg_steps_to_solution": float(info["step_count"]),
                    "deanonymization_accuracy": 0.0,
                    "avg_reward": float(info["total_reward"]),
                    "leaderboard_score": 0.0,
                },
                "episodes": [
                    {
                        "task_id": env.state.task.task_id if env.state else "n/a",
                        "task_type": env.state.task.task_type if env.state else "n/a",
                        "question": env.state.task.question if env.state else "n/a",
                        "task_answer": str(info["task_answer"]),
                        "agent_answer": str(info["agent_answer"]),
                        "graph_f1": 0.0,
                        "reward": float(info["total_reward"]),
                        "steps": int(info["step_count"]),
                        "tool_calls": int(info["tool_calls"]),
                        "success": int(info["agent_answer"] == info["task_answer"]),
                    }
                ],
            }

        graph_f1 = 0.0
        if env.state is not None:
            graph_f1 = compute_graph_f1(env.memory_graph.edges, env.state.task.supporting_edges)

        if evaluation is None:
            summary = {
                "task_success_rate": 0.0,
                "tool_efficiency": 0.0,
                "avg_graph_f1": graph_f1,
                "avg_steps_to_solution": float(env.state.step_count) if env.state else 0.0,
                "deanonymization_accuracy": 0.0,
                "avg_reward": float(env.state.total_reward) if env.state else 0.0,
                "leaderboard_score": 0.0,
            }
            evaluation = {"summary": summary, "episodes": []}

        leaderboard = load_leaderboard(leaderboard_path)
        out = export_dashboard(env=env, evaluation=evaluation, leaderboard_records=leaderboard, output_path=args.output)
        print(json.dumps({"dashboard": out, "evaluation": evaluation_path}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
