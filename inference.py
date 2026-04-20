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
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN","")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "0"))
EPISODES = int(os.getenv("EPISODES", "1"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.67"))
TASK_INDICES_RAW = os.getenv("TASK_INDICES", "")
DATASET_MODE = os.getenv("DATASET_MODE", "")
METAQA_ROOT = os.getenv("METAQA_ROOT", "")
METAQA_KB_PATH = os.getenv("METAQA_KB_PATH", "")
METAQA_VARIANT = os.getenv("METAQA_VARIANT", "")
METAQA_HOPS_RAW = os.getenv("METAQA_HOPS", "")
METAQA_SPLITS_RAW = os.getenv("METAQA_SPLITS", "")

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


def _parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


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
        f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
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

    # Inference submissions must route all calls through OpenAI-compatible client config.
    env_cfg.llm.provider = "openai"
    env_cfg.llm.model = MODEL_NAME.strip()

    if LLM_TIMEOUT_SECONDS > 0:
        env_cfg.llm.timeout_seconds = int(LLM_TIMEOUT_SECONDS)

    # Evaluation harnesses inject API_BASE_URL/HF_TOKEN for proxy-enforced requests.
    resolved_openai_base = API_BASE_URL.strip() or OPENAI_BASE_URL.strip() or HF_SPACE_URL.strip()
    if resolved_openai_base:
        env_cfg.llm.openai_base_url = _normalize_openai_base_url(resolved_openai_base)

    if HF_TOKEN.strip():
        env_cfg.llm.openai_api_key = HF_TOKEN.strip()
    elif API_KEY.strip():
        env_cfg.llm.openai_api_key = API_KEY.strip()
    elif OPENAI_API_KEY.strip():
        env_cfg.llm.openai_api_key = OPENAI_API_KEY.strip()

    if OPENAI_API_KEY_ENV.strip():
        env_cfg.llm.openai_api_key_env = OPENAI_API_KEY_ENV.strip()

    dataset_mode = DATASET_MODE.strip().lower()
    if dataset_mode in {"canonical", "metaqa"}:
        env_cfg.dataset_mode = dataset_mode

    if METAQA_ROOT.strip():
        env_cfg.metaqa_root = METAQA_ROOT.strip()
    if METAQA_KB_PATH.strip():
        env_cfg.metaqa_kb_path = METAQA_KB_PATH.strip()

    metaqa_variant = METAQA_VARIANT.strip().lower()
    if metaqa_variant in {"vanilla", "ntm"}:
        env_cfg.metaqa_variant = metaqa_variant

    metaqa_hops = _parse_csv_tokens(METAQA_HOPS_RAW)
    if metaqa_hops:
        env_cfg.metaqa_hops = metaqa_hops

    metaqa_splits = _parse_csv_tokens(METAQA_SPLITS_RAW)
    if metaqa_splits:
        env_cfg.metaqa_splits = metaqa_splits

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


def _last_action_error(observation: Any, info: dict[str, Any]) -> str | None:
    raw = info.get("last_action_error") if isinstance(info, dict) else None
    if raw is not None:
        return str(raw)

    tool_outputs = getattr(observation, "tool_outputs", None)
    if isinstance(tool_outputs, list) and tool_outputs:
        last = tool_outputs[-1]
        if isinstance(last, dict):
            output = last.get("output")
            if isinstance(output, dict) and output.get("error") is not None:
                return str(output.get("error"))
    return None


def _install_step_logger(env: OSINTEnvironment) -> tuple[list[float], dict[str, int], Any]:
    rewards: list[float] = []
    counters = {"steps": 0}
    original_step = env.step

    def _logged_step(action: Any):
        observation, reward, done, info = original_step(action)
        counters["steps"] += 1
        reward_value = float(reward or 0.0)
        rewards.append(reward_value)
        action_type = getattr(action, "action_type", "")
        action_type_value = str(getattr(action_type, "value", action_type))
        action_text = _format_action(
            {
                "action_type": action_type_value,
                "payload": dict(getattr(action, "payload", {}) or {}),
            }
        )
        log_step(
            step=counters["steps"],
            action=action_text,
            reward=reward_value,
            done=bool(done),
            error=_last_action_error(observation, info if isinstance(info, dict) else {}),
        )
        return observation, reward, done, info

    env.step = _logged_step
    return rewards, counters, original_step


def _validate_required_configuration() -> None:
    missing: list[str] = []

    api_base = API_BASE_URL.strip()
    model_name = MODEL_NAME.strip()
    hf_token = HF_TOKEN.strip()
    api_key = API_KEY.strip()
    openai_key = OPENAI_API_KEY.strip()

    if not api_base or api_base == "<your-active-endpoint>":
        missing.append("API_BASE_URL")
    if not model_name or model_name == "<your-active-model>":
        missing.append("MODEL_NAME")
    if not (hf_token or api_key or openai_key):
        missing.append("HF_TOKEN|API_KEY|OPENAI_API_KEY")

    # Required when using docker-image based env construction.
    if os.getenv("REQUIRE_LOCAL_IMAGE_NAME", "0").strip().lower() in {"1", "true", "yes", "on"}:
        if not LOCAL_IMAGE_NAME.strip():
            missing.append("LOCAL_IMAGE_NAME")

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(sorted(set(missing)))}")


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
    rewards, counters, original_step = _install_step_logger(env)

    single_runner = SingleAgentRunner(env=env, llm=llm)
    swarm_runner = SwarmAgentRunner(env=env, llm=llm) if env.config.swarm.enabled else None

    try:
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

            graph_f1 = compute_graph_f1(env.memory_graph.edges, env.state.task.supporting_edges)
            metrics.add(info, task_type=env.state.task.task_type, graph_f1=graph_f1)
            episode_rows.append(_episode_row(env, info))
    finally:
        env.step = original_step

    return metrics.summary(), episode_rows, rewards, int(counters["steps"])


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
    _validate_required_configuration()
    env_cfg = _resolve_environment_config()
    llm_client = build_llm_client(env_cfg.llm)

    episodes_given = "EPISODES" in os.environ and str(os.getenv("EPISODES", "")).strip() != ""
    task_indices_given = bool(TASK_INDICES)

    if not episodes_given and not task_indices_given:
        runs: list[tuple[str, list[int], int]] = [
            ("easy", list(range(0, 10)), 10),
            ("mid", list(range(10, 20)), 10),
            ("hard", list(range(20, 30)), 10),
        ]
    else:
        selected_indices = TASK_INDICES if task_indices_given else []
        episodes = len(selected_indices) if selected_indices else max(1, EPISODES)
        runs = [(TASK_NAME, selected_indices, episodes)]

    for task_name, run_indices, run_episodes in runs:
        env: OSINTEnvironment | None = None
        rewards: list[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        env = OSINTEnvironment(env_cfg, llm=llm_client)
        log_start(task=task_name, env=BENCHMARK, model=env_cfg.llm.model)

        try:
            summary, episode_rows, rewards, steps_taken = _run_with_runner(
                env=env,
                llm=llm_client,
                episodes=run_episodes,
                task_indices=run_indices,
            )

            score = float(summary.get("avg_reward", 0.0) or 0.0)
            score = max(0.0, min(1.0, score))
            success = score >= SUCCESS_SCORE_THRESHOLD

            _maybe_write_artifacts(
                env=env,
                summary=summary,
                episodes=run_episodes,
                episode_rows=episode_rows,
            )
        finally:
            if env is not None:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
            log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
