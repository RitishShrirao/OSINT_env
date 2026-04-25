from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class KimiGRPOPhaseConfig:
    """Configuration for one GRPO phase in the alternating self-play loop."""

    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 1e-6
    max_steps: int = 64
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.01
    epsilon: float = 0.2
    num_iterations: int = 1
    loss_type: str = "dapo"
    scale_rewards: str = "none"
    logging_steps: int = 10
    save_steps: int = 50
    output_subdir: str = "phase"
    use_vllm: bool = False
    vllm_mode: str = "colocate"


@dataclass(slots=True)
class GeneratorRewardWeights:
    """Weighted components for adversarial task-generator reward."""

    validity: float = 0.35
    hardness: float = 0.45
    diversity: float = 0.10
    consistency: float = 0.10


@dataclass(slots=True)
class LoraTuningConfig:
    """LoRA hyperparameters for parameter-efficient GRPO updates."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(slots=True)
class SwarmV2SwarmConfig:
    """Config for one orchestrated swarm role inside the swarm_v2 pipeline."""

    shared_context: bool = True
    max_agents: int = 4
    max_breadth: int = 3
    max_depth: int = 2
    planner_rounds: int = 2
    tools_per_agent: int = 2


@dataclass(slots=True)
class SwarmV2ValidationConfig:
    """Validation and replay limits for swarm_v2 task generation."""

    max_support_edges: int = 8
    max_path_hops: int = 4
    max_context_nodes: int = 14
    max_context_edges: int = 8
    duplicate_similarity_threshold: float = 0.8


@dataclass(slots=True)
class SwarmV2SharedContextConfig:
    """Shared context budgets used by both generator and answerer swarms."""

    shared_by_default: bool = True
    max_nodes: int = 14
    max_edges: int = 8
    target_pressure: float = 0.85


@dataclass(slots=True)
class SwarmV2Config:
    """Config block for the config-gated Swarm Self-Play v2 pipeline."""

    generator_swarm: SwarmV2SwarmConfig = field(default_factory=SwarmV2SwarmConfig)
    answerer_swarm: SwarmV2SwarmConfig = field(
        default_factory=lambda: SwarmV2SwarmConfig(
            shared_context=True,
            max_agents=3,
            max_breadth=2,
            max_depth=2,
            planner_rounds=2,
            tools_per_agent=2,
        )
    )
    validation: SwarmV2ValidationConfig = field(default_factory=SwarmV2ValidationConfig)
    shared_context: SwarmV2SharedContextConfig = field(default_factory=SwarmV2SharedContextConfig)


@dataclass(slots=True)
class SelfPlayTrainingConfig:
    """Top-level adversarial self-play training configuration."""

    rounds: int = 3
    output_dir: str = "artifacts/self_play"
    dry_run: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "osint-self-play"
    wandb_entity: str = ""
    wandb_run_name_prefix: str = "self-play"
    pipeline_mode: str = "legacy"
    model_topology: str = "dual"
    phase_schedule: str = "generator_answerer"
    tuning_mode: str = "full"
    shared_model_name_or_path: str = ""
    seed_tasks_per_round: int = 16
    generated_tasks_per_round: int = 24
    generator_prompts_per_round: int = 24
    max_graph_context_nodes: int = 100
    max_graph_context_edges: int = 100
    max_support_edges: int = 8
    answerer_judge_max_new_tokens: int = 48
    generator_reward_weights: GeneratorRewardWeights = field(default_factory=GeneratorRewardWeights)
    lora: LoraTuningConfig = field(default_factory=LoraTuningConfig)
    swarm_v2: SwarmV2Config = field(default_factory=SwarmV2Config)
    generator_phase: KimiGRPOPhaseConfig = field(
        default_factory=lambda: KimiGRPOPhaseConfig(output_subdir="generator")
    )
    answerer_phase: KimiGRPOPhaseConfig = field(
        default_factory=lambda: KimiGRPOPhaseConfig(output_subdir="answerer")
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_int(value: Any, default: int, floor: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if floor is not None:
        out = max(floor, out)
    return out


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _parse_str_choice(value: Any, default: str, allowed: set[str]) -> str:
    token = str(value).strip().lower()
    if token in allowed:
        return token
    return default


def _parse_str_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or list(fallback)
    if isinstance(value, str):
        out = [item.strip() for item in value.split(",") if item.strip()]
        return out or list(fallback)
    return list(fallback)


def _parse_phase(data: dict[str, Any], fallback: KimiGRPOPhaseConfig) -> KimiGRPOPhaseConfig:
    return KimiGRPOPhaseConfig(
        model_name_or_path=str(data.get("model_name_or_path", fallback.model_name_or_path)).strip()
        or fallback.model_name_or_path,
        learning_rate=_parse_float(data.get("learning_rate"), fallback.learning_rate),
        max_steps=_parse_int(data.get("max_steps"), fallback.max_steps, floor=1),
        per_device_train_batch_size=_parse_int(
            data.get("per_device_train_batch_size"),
            fallback.per_device_train_batch_size,
            floor=1,
        ),
        gradient_accumulation_steps=_parse_int(
            data.get("gradient_accumulation_steps"),
            fallback.gradient_accumulation_steps,
            floor=1,
        ),
        num_generations=_parse_int(data.get("num_generations"), fallback.num_generations, floor=1),
        max_completion_length=_parse_int(
            data.get("max_completion_length"),
            fallback.max_completion_length,
            floor=1,
        ),
        temperature=_parse_float(data.get("temperature"), fallback.temperature),
        top_p=_parse_float(data.get("top_p"), fallback.top_p),
        beta=_parse_float(data.get("beta"), fallback.beta),
        epsilon=_parse_float(data.get("epsilon"), fallback.epsilon),
        num_iterations=_parse_int(data.get("num_iterations"), fallback.num_iterations, floor=1),
        loss_type=str(data.get("loss_type", fallback.loss_type)).strip() or fallback.loss_type,
        scale_rewards=str(data.get("scale_rewards", fallback.scale_rewards)).strip() or fallback.scale_rewards,
        logging_steps=_parse_int(data.get("logging_steps"), fallback.logging_steps, floor=1),
        save_steps=_parse_int(data.get("save_steps"), fallback.save_steps, floor=1),
        output_subdir=str(data.get("output_subdir", fallback.output_subdir)).strip() or fallback.output_subdir,
        use_vllm=_parse_bool(data.get("use_vllm"), fallback.use_vllm),
        vllm_mode=str(data.get("vllm_mode", fallback.vllm_mode)).strip() or fallback.vllm_mode,
    )


def _parse_generator_weights(data: dict[str, Any]) -> GeneratorRewardWeights:
    return GeneratorRewardWeights(
        validity=_parse_float(data.get("validity"), 0.35),
        hardness=_parse_float(data.get("hardness"), 0.45),
        diversity=_parse_float(data.get("diversity"), 0.10),
        consistency=_parse_float(data.get("consistency"), 0.10),
    )


def _parse_lora_config(data: dict[str, Any], fallback: LoraTuningConfig) -> LoraTuningConfig:
    return LoraTuningConfig(
        r=_parse_int(data.get("r"), fallback.r, floor=1),
        alpha=_parse_int(data.get("alpha"), fallback.alpha, floor=1),
        dropout=_parse_float(data.get("dropout"), fallback.dropout),
        target_modules=_parse_str_list(data.get("target_modules"), fallback.target_modules),
        bias=str(data.get("bias", fallback.bias)).strip() or fallback.bias,
        task_type=str(data.get("task_type", fallback.task_type)).strip() or fallback.task_type,
    )


def _parse_swarm_v2_swarm_config(
    data: dict[str, Any],
    fallback: SwarmV2SwarmConfig,
) -> SwarmV2SwarmConfig:
    return SwarmV2SwarmConfig(
        shared_context=_parse_bool(data.get("shared_context"), fallback.shared_context),
        max_agents=_parse_int(data.get("max_agents"), fallback.max_agents, floor=1),
        max_breadth=_parse_int(data.get("max_breadth"), fallback.max_breadth, floor=1),
        max_depth=_parse_int(data.get("max_depth"), fallback.max_depth, floor=1),
        planner_rounds=_parse_int(data.get("planner_rounds"), fallback.planner_rounds, floor=1),
        tools_per_agent=_parse_int(data.get("tools_per_agent"), fallback.tools_per_agent, floor=1),
    )


def _parse_swarm_v2_validation_config(
    data: dict[str, Any],
    fallback: SwarmV2ValidationConfig,
    legacy_max_support_edges: int,
) -> SwarmV2ValidationConfig:
    default_max_support_edges = (
        _parse_int(data.get("max_support_edges"), legacy_max_support_edges, floor=1)
        if "max_support_edges" not in data
        else _parse_int(data.get("max_support_edges"), fallback.max_support_edges, floor=1)
    )
    return SwarmV2ValidationConfig(
        max_support_edges=default_max_support_edges,
        max_path_hops=_parse_int(data.get("max_path_hops"), fallback.max_path_hops, floor=1),
        max_context_nodes=_parse_int(data.get("max_context_nodes"), fallback.max_context_nodes, floor=1),
        max_context_edges=_parse_int(data.get("max_context_edges"), fallback.max_context_edges, floor=1),
        duplicate_similarity_threshold=max(
            0.0,
            min(
                1.0,
                _parse_float(
                    data.get("duplicate_similarity_threshold"),
                    fallback.duplicate_similarity_threshold,
                ),
            ),
        ),
    )


def _parse_swarm_v2_shared_context_config(
    data: dict[str, Any],
    fallback: SwarmV2SharedContextConfig,
) -> SwarmV2SharedContextConfig:
    return SwarmV2SharedContextConfig(
        shared_by_default=_parse_bool(data.get("shared_by_default"), fallback.shared_by_default),
        max_nodes=_parse_int(data.get("max_nodes"), fallback.max_nodes, floor=1),
        max_edges=_parse_int(data.get("max_edges"), fallback.max_edges, floor=1),
        target_pressure=max(0.0, min(1.0, _parse_float(data.get("target_pressure"), fallback.target_pressure))),
    )


def _parse_swarm_v2_config(
    data: dict[str, Any],
    fallback: SwarmV2Config,
    legacy_max_support_edges: int,
) -> SwarmV2Config:
    return SwarmV2Config(
        generator_swarm=_parse_swarm_v2_swarm_config(
            _as_dict(data.get("generator_swarm")),
            fallback.generator_swarm,
        ),
        answerer_swarm=_parse_swarm_v2_swarm_config(
            _as_dict(data.get("answerer_swarm")),
            fallback.answerer_swarm,
        ),
        validation=_parse_swarm_v2_validation_config(
            _as_dict(data.get("validation")),
            fallback.validation,
            legacy_max_support_edges=legacy_max_support_edges,
        ),
        shared_context=_parse_swarm_v2_shared_context_config(
            _as_dict(data.get("shared_context")),
            fallback.shared_context,
        ),
    )


def load_self_play_config(path: str | Path | None) -> SelfPlayTrainingConfig:
    if not path:
        return SelfPlayTrainingConfig()

    file_path = Path(path)
    if not file_path.exists():
        return SelfPlayTrainingConfig()

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Self-play config file must contain a JSON object.")

    defaults = SelfPlayTrainingConfig()
    generator_phase = _parse_phase(_as_dict(payload.get("generator_phase")), defaults.generator_phase)
    answerer_phase = _parse_phase(_as_dict(payload.get("answerer_phase")), defaults.answerer_phase)
    lora_cfg = _parse_lora_config(_as_dict(payload.get("lora")), defaults.lora)
    legacy_max_support_edges = _parse_int(payload.get("max_support_edges"), defaults.max_support_edges, floor=1)
    swarm_v2_cfg = _parse_swarm_v2_config(
        _as_dict(payload.get("swarm_v2")),
        defaults.swarm_v2,
        legacy_max_support_edges=legacy_max_support_edges,
    )

    return SelfPlayTrainingConfig(
        rounds=_parse_int(payload.get("rounds"), defaults.rounds, floor=1),
        output_dir=str(payload.get("output_dir", defaults.output_dir)).strip() or defaults.output_dir,
        dry_run=_parse_bool(payload.get("dry_run"), defaults.dry_run),
        wandb_enabled=_parse_bool(payload.get("wandb_enabled"), defaults.wandb_enabled),
        wandb_project=str(payload.get("wandb_project", defaults.wandb_project)).strip() or defaults.wandb_project,
        wandb_entity=str(payload.get("wandb_entity", defaults.wandb_entity)).strip(),
        wandb_run_name_prefix=str(payload.get("wandb_run_name_prefix", defaults.wandb_run_name_prefix)).strip()
        or defaults.wandb_run_name_prefix,
        pipeline_mode=_parse_str_choice(
            payload.get("pipeline_mode"),
            defaults.pipeline_mode,
            {"legacy", "swarm_v2"},
        ),
        model_topology=_parse_str_choice(
            payload.get("model_topology"),
            defaults.model_topology,
            {"dual", "shared"},
        ),
        phase_schedule=_parse_str_choice(
            payload.get("phase_schedule"),
            defaults.phase_schedule,
            {"generator_answerer", "answerer_generator_answerer"},
        ),
        tuning_mode=_parse_str_choice(
            payload.get("tuning_mode"),
            defaults.tuning_mode,
            {"full", "lora"},
        ),
        shared_model_name_or_path=str(
            payload.get("shared_model_name_or_path", defaults.shared_model_name_or_path)
        ).strip(),
        seed_tasks_per_round=_parse_int(
            payload.get("seed_tasks_per_round"),
            defaults.seed_tasks_per_round,
            floor=1,
        ),
        generated_tasks_per_round=_parse_int(
            payload.get("generated_tasks_per_round"),
            defaults.generated_tasks_per_round,
            floor=1,
        ),
        generator_prompts_per_round=_parse_int(
            payload.get("generator_prompts_per_round"),
            defaults.generator_prompts_per_round,
            floor=1,
        ),
        max_graph_context_nodes=_parse_int(
            payload.get("max_graph_context_nodes"),
            defaults.max_graph_context_nodes,
            floor=1,
        ),
        max_graph_context_edges=_parse_int(
            payload.get("max_graph_context_edges"),
            defaults.max_graph_context_edges,
            floor=1,
        ),
        max_support_edges=legacy_max_support_edges,
        answerer_judge_max_new_tokens=_parse_int(
            payload.get("answerer_judge_max_new_tokens"),
            defaults.answerer_judge_max_new_tokens,
            floor=1,
        ),
        generator_reward_weights=_parse_generator_weights(
            _as_dict(payload.get("generator_reward_weights"))
        ),
        lora=lora_cfg,
        swarm_v2=swarm_v2_cfg,
        generator_phase=generator_phase,
        answerer_phase=answerer_phase,
    )
