from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osint_env.domain.models import (
    EnvironmentConfig,
    NodeType,
    SeedingConfig,
    SeedEdgeSpec,
    SeedNodeSpec,
    SeedQuestionSpec,
    SpawnRewardConfig,
    SwarmConfig,
)


@dataclass(slots=True)
class RuntimeDefaults:
    default_episodes: int = 20
    leaderboard_path: str = "artifacts/leaderboard.json"
    dashboard_path: str = "artifacts/osint_dashboard.html"
    sweep_dashboard_dir: str = "artifacts/sweep_dashboards"


@dataclass(slots=True)
class SharedConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    runtime: RuntimeDefaults = field(default_factory=RuntimeDefaults)


def clone_environment_config(config: EnvironmentConfig) -> EnvironmentConfig:
    return copy.deepcopy(config)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _infer_node_type(node_id: str) -> NodeType:
    prefix = str(node_id).split("_", 1)[0].lower()
    mapping = {
        "user": NodeType.USER,
        "alias": NodeType.ALIAS,
        "org": NodeType.ORG,
        "loc": NodeType.LOCATION,
        "location": NodeType.LOCATION,
        "post": NodeType.POST,
        "thr": NodeType.THREAD,
        "thread": NodeType.THREAD,
        "event": NodeType.EVENT,
    }
    return mapping.get(prefix, NodeType.USER)


def _parse_node_type(value: Any, node_id: str) -> NodeType:
    if isinstance(value, NodeType):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        try:
            return NodeType(raw)
        except ValueError:
            return _infer_node_type(node_id)
    return _infer_node_type(node_id)


def _parse_seed_edge(item: dict[str, Any]) -> SeedEdgeSpec | None:
    src = str(item.get("src", "")).strip()
    rel = str(item.get("rel", "")).strip()
    dst = str(item.get("dst", "")).strip()
    if not src or not rel or not dst:
        return None
    confidence = _parse_float(item.get("confidence", 1.0), 1.0)
    return SeedEdgeSpec(src=src, rel=rel, dst=dst, confidence=confidence)


def _parse_seeding(data: dict[str, Any]) -> SeedingConfig:
    seeded_nodes: list[SeedNodeSpec] = []
    for item in data.get("seeded_nodes", []):
        row = _as_dict(item)
        node_id = str(row.get("node_id", "")).strip()
        if not node_id:
            continue
        node_type = _parse_node_type(row.get("node_type"), node_id)
        attrs = _as_dict(row.get("attrs"))
        seeded_nodes.append(SeedNodeSpec(node_id=node_id, node_type=node_type, attrs=attrs))

    seeded_edges: list[SeedEdgeSpec] = []
    for item in data.get("seeded_edges", []):
        edge = _parse_seed_edge(_as_dict(item))
        if edge is not None:
            seeded_edges.append(edge)

    seeded_questions: list[SeedQuestionSpec] = []
    for item in data.get("seeded_questions", []):
        row = _as_dict(item)
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        answer_val = row.get("answer")
        answer = str(answer_val).strip() if answer_val is not None and str(answer_val).strip() else None
        task_type = str(row.get("task_type", "seeded")).strip() or "seeded"
        support_edges: list[SeedEdgeSpec] = []
        for edge_item in row.get("supporting_edges", []):
            edge = _parse_seed_edge(_as_dict(edge_item))
            if edge is not None:
                support_edges.append(edge)
        metadata = _as_dict(row.get("metadata"))
        seeded_questions.append(
            SeedQuestionSpec(
                question=question,
                answer=answer,
                task_type=task_type,
                supporting_edges=support_edges,
                metadata=metadata,
            )
        )

    return SeedingConfig(
        seeded_nodes=seeded_nodes,
        seeded_edges=seeded_edges,
        seeded_questions=seeded_questions,
        llm_generate_remaining_graph=_parse_bool(data.get("llm_generate_remaining_graph"), True),
        llm_generate_remaining_tasks=_parse_bool(data.get("llm_generate_remaining_tasks"), True),
        llm_generated_edge_budget=max(0, _parse_int(data.get("llm_generated_edge_budget"), 6)),
        llm_generated_task_budget=max(0, _parse_int(data.get("llm_generated_task_budget"), 8)),
    )


def load_seeding_config(path: str | Path) -> SeedingConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Seed file must contain a JSON object.")
    source = _as_dict(payload.get("seeding", payload))
    return _parse_seeding(source)


def _parse_environment(payload: dict[str, Any]) -> EnvironmentConfig:
    env_data = _as_dict(payload.get("environment", payload))
    swarm_data = _as_dict(payload.get("swarm", env_data.get("swarm", {})))
    spawn_data = _as_dict(payload.get("spawn_reward", env_data.get("spawn_reward", {})))
    seeding_data = _as_dict(payload.get("seeding", env_data.get("seeding", {})))

    env = EnvironmentConfig(
        n_users=max(4, _parse_int(env_data.get("n_users"), 40)),
        alias_density=max(0.0, min(1.0, _parse_float(env_data.get("alias_density"), 0.35))),
        noise_level=max(0.0, min(1.0, _parse_float(env_data.get("noise_level"), 0.15))),
        red_herring_rate=max(0.0, min(1.0, _parse_float(env_data.get("red_herring_rate"), 0.1))),
        max_steps=max(2, _parse_int(env_data.get("max_steps"), 18)),
        seed=_parse_int(env_data.get("seed"), 7),
    )

    env.swarm = SwarmConfig(
        enabled=_parse_bool(swarm_data.get("enabled"), False),
        max_agents=max(1, _parse_int(swarm_data.get("max_agents"), 3)),
        max_breadth=max(1, _parse_int(swarm_data.get("max_breadth"), 2)),
        max_width=max(1, _parse_int(swarm_data.get("max_width"), 2)),
        max_depth=max(1, _parse_int(swarm_data.get("max_depth"), 2)),
        planner_rounds=max(1, _parse_int(swarm_data.get("planner_rounds"), 2)),
        tools_per_agent=max(1, _parse_int(swarm_data.get("tools_per_agent"), 1)),
    )

    env.spawn_reward = SpawnRewardConfig(
        lambda_parallel=max(0.0, _parse_float(spawn_data.get("lambda_parallel"), 0.15)),
        lambda_finish=max(0.0, _parse_float(spawn_data.get("lambda_finish"), 0.2)),
        anneal=max(0.0, min(1.0, _parse_float(spawn_data.get("anneal"), 1.0))),
        max_parallel_hint=max(1, _parse_int(spawn_data.get("max_parallel_hint"), 3)),
    )

    env.seeding = _parse_seeding(seeding_data)
    return env


def _parse_runtime(payload: dict[str, Any]) -> RuntimeDefaults:
    runtime = _as_dict(payload.get("runtime", {}))
    return RuntimeDefaults(
        default_episodes=max(1, _parse_int(runtime.get("default_episodes"), 20)),
        leaderboard_path=str(runtime.get("leaderboard_path", "artifacts/leaderboard.json")),
        dashboard_path=str(runtime.get("dashboard_path", "artifacts/osint_dashboard.html")),
        sweep_dashboard_dir=str(runtime.get("sweep_dashboard_dir", "artifacts/sweep_dashboards")),
    )


def load_shared_config(path: str | Path | None) -> SharedConfig:
    if not path:
        return SharedConfig()

    file_path = Path(path)
    if not file_path.exists():
        return SharedConfig()

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Shared config file must contain a JSON object.")

    return SharedConfig(environment=_parse_environment(payload), runtime=_parse_runtime(payload))
