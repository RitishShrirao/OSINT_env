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
class SelfPlayTrainingConfig:
    """Top-level adversarial self-play training configuration."""

    rounds: int = 3
    output_dir: str = "artifacts/self_play"
    dry_run: bool = True
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

    return SelfPlayTrainingConfig(
        rounds=_parse_int(payload.get("rounds"), defaults.rounds, floor=1),
        output_dir=str(payload.get("output_dir", defaults.output_dir)).strip() or defaults.output_dir,
        dry_run=_parse_bool(payload.get("dry_run"), defaults.dry_run),
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
        max_support_edges=_parse_int(payload.get("max_support_edges"), defaults.max_support_edges, floor=1),
        answerer_judge_max_new_tokens=_parse_int(
            payload.get("answerer_judge_max_new_tokens"),
            defaults.answerer_judge_max_new_tokens,
            floor=1,
        ),
        generator_reward_weights=_parse_generator_weights(
            _as_dict(payload.get("generator_reward_weights"))
        ),
        lora=lora_cfg,
        generator_phase=generator_phase,
        answerer_phase=answerer_phase,
    )
