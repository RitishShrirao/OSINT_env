from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from osint_env.agents.single_agent import SingleAgentRunner
from osint_env.agents.swarm_agent import SwarmAgentRunner
from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.domain.models import EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.reward import compute_graph_f1
from osint_env.eval.leaderboard import append_leaderboard_record, load_leaderboard
from osint_env.eval.metrics import EvalMetrics
from osint_env.llm import build_llm_client
from osint_env.viz import export_dashboard


CONFIG_PATH = os.getenv("CONFIG_PATH", "datasets/fixed_levels/shared_config_fixed_levels.json")
SEED_FILE = os.getenv("SEED_FILE", "datasets/fixed_levels/seed_fixed_levels.json")
AGENT_MODE = os.getenv("AGENT_MODE", "swarm")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "0"))
EPISODES = int(os.getenv("EPISODES", "1"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.67"))
TASK_INDICES_RAW = os.getenv("TASK_INDICES", "")

WRITE_BENCHMARK_ARTIFACTS = os.getenv("WRITE_BENCHMARK_ARTIFACTS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LEADERBOARD_PATH = os.getenv("LEADERBOARD_PATH", "datasets/fixed_levels/leaderboard_fixed_levels.json")
DASHBOARD_PATH = os.getenv("DASHBOARD_PATH", "datasets/fixed_levels/dashboard_fixed_levels.html")
RUN_NAME = os.getenv("RUN_NAME", "fixed_levels_qwen_swarm")

BENCHMARK = "osint-openenv"
TASK_NAME = "fixed_levels_easy_mid_hard"


def _parse_task_indices(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw or "").split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            out.append(int(stripped))
        except ValueError:
            continue
    return out


def _normalize_ollama_base_url(url: str) -> str:
    normalized = str(url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized or "http://127.0.0.1:11434"


def _normalize_openai_base_url(url: str) -> str:
    normalized = str(url or "").strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


TASK_INDICES = _parse_task_indices(TASK_INDICES_RAW)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(bool(done)).lower()} error={error_text}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] task={task} success={str(bool(success)).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def _looks_like_placeholder_api_key(value: str) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return True
    placeholder_markers = [
        "your_openai_api_key",
        "your-key",
        "your_key",
        "your real",
        "real-openai-key",
        "replace-me",
        "changeme",
        "example",
        "<api-key>",
    ]
    if token.startswith("your_") or token.startswith("sk-your-"):
        return True
    return any(marker in token for marker in placeholder_markers)


def _format_action(action: dict[str, Any]) -> str:
    action_type = str(action.get("action_type", "")).upper()
    payload = dict(action.get("payload", {}))

    if action_type == "ANSWER":
        return f"answer({str(payload.get('answer', 'unknown')).strip()})"

    if action_type == "ADD_EDGE":
        try:
            conf = float(payload.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        return (
            "add_edge("
            f"{payload.get('src', '')},"
            f"{payload.get('rel', '')},"
            f"{payload.get('dst', '')},"
            f"{conf:.2f}"
            ")"
        )

    tool_name = str(payload.get("tool_name", "tool")).strip() or "tool"
    args = payload.get("args", {})
    if not isinstance(args, dict) or not args:
        return f"{tool_name}()"
    args_text = ",".join(f"{key}={value}" for key, value in sorted(args.items()))
    return f"{tool_name}({args_text})"


def _assistant_tool_call_id(message: dict[str, Any]) -> str | None:
    tool_calls = list(message.get("tool_calls", []))
    if not tool_calls:
        return None
    tool_call_id = tool_calls[0].get("id")
    return str(tool_call_id) if tool_call_id else None


def _tool_result_message(assistant_message: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    tool_call_id = _assistant_tool_call_id(assistant_message)
    if not tool_call_id:
        return None
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, sort_keys=True),
    }


def _resolve_environment_config() -> EnvironmentConfig:
    shared = load_shared_config(CONFIG_PATH)
    env_cfg = clone_environment_config(shared.environment)

    if SEED_FILE and Path(SEED_FILE).exists():
        env_cfg.seeding = load_seeding_config(SEED_FILE)

    mode = AGENT_MODE.strip().lower()
    if mode == "single":
        env_cfg.swarm.enabled = False
    elif mode == "swarm":
        env_cfg.swarm.enabled = True

    provider = LLM_PROVIDER.strip().lower()
    if provider and provider != "config":
        env_cfg.llm.provider = provider

    if MODEL_NAME.strip():
        env_cfg.llm.model = MODEL_NAME.strip()

    if LLM_TIMEOUT_SECONDS > 0:
        env_cfg.llm.timeout_seconds = int(LLM_TIMEOUT_SECONDS)

    if provider == "openai":
        # Evaluation harnesses often inject API_BASE_URL/API_KEY for proxy enforcement.
        resolved_openai_base = API_BASE_URL.strip() or OPENAI_BASE_URL.strip() or HF_SPACE_URL.strip()
        if resolved_openai_base:
            env_cfg.llm.openai_base_url = _normalize_openai_base_url(resolved_openai_base)

        if API_KEY.strip():
            env_cfg.llm.openai_api_key = API_KEY.strip()
        elif OPENAI_API_KEY.strip():
            env_cfg.llm.openai_api_key = OPENAI_API_KEY.strip()
        elif HF_TOKEN.strip():
            env_cfg.llm.openai_api_key = HF_TOKEN.strip()
    elif API_BASE_URL.strip() or OLLAMA_BASE_URL.strip():
        env_cfg.llm.ollama_base_url = _normalize_ollama_base_url(API_BASE_URL or OLLAMA_BASE_URL)

    if OPENAI_API_KEY_ENV.strip():
        env_cfg.llm.openai_api_key_env = OPENAI_API_KEY_ENV.strip()

    return env_cfg


def _runner_for(env: OSINTEnvironment, llm: Any) -> SingleAgentRunner | SwarmAgentRunner:
    if env.config.swarm.enabled:
        return SwarmAgentRunner(env=env, llm=llm)
    return SingleAgentRunner(env=env, llm=llm)


def _normalize_difficulty(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in {"easy", "e"}:
        return "easy"
    if token in {"mid", "medium", "m"}:
        return "medium"
    if token in {"high", "hard", "h"}:
        return "hard"
    return "hard"


def _task_difficulty(env: OSINTEnvironment, task_index: int) -> str:
    idx = int(task_index) % max(1, len(env.tasks))
    task = env.tasks[idx]
    if isinstance(task.metadata, dict) and "difficulty" in task.metadata:
        return _normalize_difficulty(str(task.metadata.get("difficulty", "")))
    if idx < 10:
        return "easy"
    if idx < 20:
        return "medium"
    return "hard"


def _episode_row(env: OSINTEnvironment, info: dict[str, Any]) -> dict[str, Any]:
    if env.state is None:
        return {
            "task_id": "unknown",
            "task_type": "unknown",
            "question": "",
            "task_answer": str(info.get("task_answer", "")),
            "agent_answer": str(info.get("agent_answer", "")),
            "graph_f1": 0.0,
            "reward": float(info.get("total_reward", 0.0) or 0.0),
            "steps": int(info.get("step_count", 0) or 0),
            "tool_calls": int(info.get("tool_calls", 0) or 0),
            "success": int(info.get("agent_answer") == info.get("task_answer")),
            "reward_components": dict(info.get("reward_components", {})),
            "pred_edges": [],
            "truth_edges": [],
        }

    graph_f1 = compute_graph_f1(env.memory_graph.edges, env.state.task.supporting_edges)
    return {
        "task_id": env.state.task.task_id,
        "task_type": env.state.task.task_type,
        "question": env.state.task.question,
        "task_answer": str(info.get("task_answer", "")),
        "agent_answer": str(info.get("agent_answer", "")) if info.get("agent_answer") is not None else "",
        "graph_f1": graph_f1,
        "reward": float(info.get("total_reward", 0.0) or 0.0),
        "steps": int(info.get("step_count", 0) or 0),
        "tool_calls": int(info.get("tool_calls", 0) or 0),
        "success": int(info.get("agent_answer") == info.get("task_answer")),
        "reward_components": dict(info.get("reward_components", {})),
        "spawn_count": int(info.get("spawn_count", 0) or 0),
        "spawn_critical_steps": int(info.get("spawn_critical_steps", 0) or 0),
        "pred_edges": [
            {
                "src": edge.src,
                "rel": edge.rel,
                "dst": edge.dst,
                "confidence": float(edge.confidence),
            }
            for edge in env.memory_graph.edges
        ],
        "truth_edges": [
            {
                "src": edge.src,
                "rel": edge.rel,
                "dst": edge.dst,
                "confidence": float(edge.confidence),
            }
            for edge in env.state.task.supporting_edges
        ],
    }


def _format_action_from_history(item: dict[str, Any]) -> str:
    action_type = str(item.get("type", "")).upper() 
    payload = dict(item.get("payload", {}))

    if action_type == "ANSWER":
        return f"answer({str(payload.get('answer', 'unknown')).strip()})"

    if action_type == "ADD_EDGE":
        try:
            conf = float(payload.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        return (
            "add_edge("
            f"{payload.get('src', '')},"
            f"{payload.get('rel', '')},"
            f"{payload.get('dst', '')},"
            f"{conf:.2f}"
            ")"
        )

    tool_name = str(payload.get("tool_name", "tool")).strip() or "tool"
    args = payload.get("args", {})
    if not isinstance(args, dict) or not args:
        return f"{tool_name}()"
    args_text = ",".join(f"{key}={value}" for key, value in sorted(args.items()))
    return f"{tool_name}({args_text})"


def _task_targets(env: OSINTEnvironment, episodes: int, task_indices: list[int]) -> list[int | None]:
    if task_indices:
        task_count = max(1, len(env.tasks))
        return [index % task_count for index in task_indices]
    return [None] * max(1, episodes)


def _run_with_runner(
    env: OSINTEnvironment,
    llm: Any,
    episodes: int,
    task_indices: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float], int]:
    metrics = EvalMetrics()
    episode_rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    steps_taken = 0

    single_runner = SingleAgentRunner(env=env, llm=llm)
    swarm_runner = SwarmAgentRunner(env=env, llm=llm) if env.config.swarm.enabled else None

    for task_index in _task_targets(env, episodes, task_indices):
        task_count = max(1, len(env.tasks))
        selected_index = env._task_idx % task_count if task_index is None else int(task_index) % task_count
        if task_index is not None:
            # Keep compatibility with explicit task selection from the previous inference script.
            env._task_idx = selected_index

        difficulty = _task_difficulty(env, selected_index)
        if difficulty == "easy":
            runner: SingleAgentRunner | SwarmAgentRunner = single_runner
        elif swarm_runner is not None:
            runner = swarm_runner
        else:
            runner = single_runner

        info = runner.run_episode()
        if env.state is None:
            continue

        history = list(env.state.action_history)
        for idx, action_item in enumerate(history, start=1):
            reward = float(action_item.get("reward", 0.0) or 0.0)
            rewards.append(reward)
            steps_taken += 1
            done = idx == len(history)
            log_step(
                step=steps_taken,
                action=_format_action_from_history(action_item),
                reward=reward,
                done=done,
                error=None,
            )

        graph_f1 = compute_graph_f1(env.memory_graph.edges, env.state.task.supporting_edges)
        metrics.add(info, task_type=env.state.task.task_type, graph_f1=graph_f1)
        episode_rows.append(_episode_row(env, info))

    return metrics.summary(), episode_rows, rewards, steps_taken


def _maybe_write_artifacts(
    env: OSINTEnvironment,
    summary: dict[str, Any],
    episodes: int,
    episode_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    if not WRITE_BENCHMARK_ARTIFACTS:
        return None, None

    record = append_leaderboard_record(
        path=LEADERBOARD_PATH,
        summary=summary,
        episodes=episodes,
        run_name=RUN_NAME or None,
        config={
            "seed": env.config.seed,
            "max_steps": env.config.max_steps,
            "swarm_enabled": env.config.swarm.enabled,
            "max_agents": env.config.swarm.max_agents,
            "max_breadth": env.config.swarm.max_breadth,
            "max_width": env.config.swarm.max_width,
            "max_depth": env.config.swarm.max_depth,
            "seeded_questions": len(env.config.seeding.seeded_questions),
            "llm_provider": env.config.llm.provider,
            "llm_model": env.config.llm.model,
        },
    )

    leaderboard = load_leaderboard(LEADERBOARD_PATH)
    dashboard = export_dashboard(
        env=env,
        evaluation={"summary": summary, "episodes": episode_rows},
        leaderboard_records=leaderboard,
        output_path=DASHBOARD_PATH,
    )
    return record, dashboard


def main() -> None:
    env_cfg = _resolve_environment_config()
    llm_client = build_llm_client(env_cfg.llm)
    env = OSINTEnvironment(env_cfg, llm=llm_client)

    log_start(task=TASK_NAME, env=BENCHMARK, model=env_cfg.llm.model)

    episodes = len(TASK_INDICES) if TASK_INDICES else max(1, EPISODES)
    summary, episode_rows, rewards, steps_taken = _run_with_runner(
        env=env,
        llm=llm_client,
        episodes=episodes,
        task_indices=TASK_INDICES,
    )

    score = float(summary.get("avg_reward", 0.0) or 0.0)
    score = max(0.01, min(0.99, score))
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(task=TASK_NAME, success=success, steps=steps_taken, score=score, rewards=rewards)

    record, dashboard = _maybe_write_artifacts(
        env=env,
        summary=summary,
        episodes=episodes,
        episode_rows=episode_rows,
    )

    payload: dict[str, Any] = {
        "summary": summary,
        "episodes": episode_rows,
    }
    if record is not None:
        payload["record"] = record
    if dashboard is not None:
        payload["dashboard"] = dashboard

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
