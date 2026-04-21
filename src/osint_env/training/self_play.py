from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from osint_env.domain.models import Edge, EnvironmentConfig, TaskInstance
from osint_env.env.environment import OSINTEnvironment
from osint_env.llm import build_llm_client
from osint_env.training.config import KimiGRPOPhaseConfig, LoraTuningConfig, SelfPlayTrainingConfig
from osint_env.training.rewards import (
    AnswererJudge,
    AnswererRewardFunction,
    GeneratorRewardFunction,
    decode_completion_text,
    parse_generated_task_completion,
)


@dataclass(slots=True)
class _RoundArtifacts:
    round_index: int
    generator_dataset_path: str
    answerer_dataset_path: str
    generated_tasks_path: str



def _require_training_stack() -> tuple[Any, Any, Any]:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training stack is missing. Install train dependencies first: "
            "python -m pip install -e .[train]"
        ) from exc
    return Dataset, GRPOConfig, GRPOTrainer



def _task_to_edge_json(task: TaskInstance) -> str:
    payload = [
        {
            "src": edge.src,
            "rel": edge.rel,
            "dst": edge.dst,
            "confidence": float(edge.confidence),
        }
        for edge in task.supporting_edges
    ]
    return json.dumps(payload, sort_keys=True)



def _difficulty_for_task(task: TaskInstance) -> str:
    metadata = dict(task.metadata or {})
    token = str(metadata.get("difficulty", "")).strip().lower()
    if token in {"easy", "medium", "hard"}:
        return token
    if task.task_type.startswith("metaqa_1-hop"):
        return "easy"
    if task.task_type.startswith("metaqa_2-hop"):
        return "medium"
    return "hard"



def _answer_prompt(question: str) -> str:
    return (
        "You are the answer-generation swarm for an OSINT graph task.\n"
        "Return only JSON in the form {\"answer\": \"<entity_or_value>\"}.\n"
        "Do not add extra keys.\n"
        f"Question: {question}"
    )



def _build_answerer_rows(tasks: list[TaskInstance]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append(
            {
                "prompt": _answer_prompt(task.question),
                "question": task.question,
                "answer": str(task.answer),
                "supporting_edges_json": _task_to_edge_json(task),
                "difficulty": _difficulty_for_task(task),
                "task_type": task.task_type,
                "task_id": task.task_id,
            }
        )
    return rows



def _graph_context_for_prompt(
    env: OSINTEnvironment,
    max_nodes: int,
    max_edges: int,
    rng: random.Random,
) -> dict[str, Any]:
    node_ids = sorted(env.graph.nodes.keys())
    if len(node_ids) > max_nodes:
        node_ids = rng.sample(node_ids, k=max_nodes)

    edges = list(env.graph.edges)
    if len(edges) > max_edges:
        edges = rng.sample(edges, k=max_edges)

    return {
        "nodes": node_ids,
        "edges": [
            {
                "src": edge.src,
                "rel": edge.rel,
                "dst": edge.dst,
            }
            for edge in edges
        ],
    }



def _generator_prompt(context_blob: dict[str, Any], anchor_questions: list[str]) -> str:
    anchors = "\n".join(f"- {question}" for question in anchor_questions)
    return (
        "You are the adversarial question-and-graph generation swarm in self-play.\n"
        "Generate one challenging but answerable OSINT task that makes answering difficult.\n"
        "Use only entities and relations from the provided graph context.\n"
        "Prefer multi-hop traces and avoid duplicates of the anchor questions.\n"
        "Return strict JSON with keys: question, answer, task_type, supporting_edges.\n"
        "supporting_edges must be a list of objects with src, rel, dst, confidence.\n"
        "Graph context:\n"
        f"{json.dumps(context_blob, sort_keys=True)}\n"
        "Anchor questions to avoid:\n"
        f"{anchors}\n"
    )



def _build_generator_rows(
    env: OSINTEnvironment,
    cfg: SelfPlayTrainingConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    existing_questions = [task.question for task in env.tasks]

    for _ in range(max(1, cfg.generator_prompts_per_round)):
        context_blob = _graph_context_for_prompt(
            env=env,
            max_nodes=cfg.max_graph_context_nodes,
            max_edges=cfg.max_graph_context_edges,
            rng=rng,
        )
        anchor_sample_size = min(5, len(existing_questions))
        anchor_sample = rng.sample(existing_questions, k=anchor_sample_size) if anchor_sample_size > 0 else []
        rows.append(
            {
                "prompt": _generator_prompt(context_blob, anchor_sample),
            }
        )
    return rows



def _safe_build_grpo_config(
    phase: KimiGRPOPhaseConfig,
    output_dir: str,
    grpo_config_cls: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": float(phase.learning_rate),
        "max_steps": int(phase.max_steps),
        "per_device_train_batch_size": int(phase.per_device_train_batch_size),
        "gradient_accumulation_steps": int(phase.gradient_accumulation_steps),
        "num_generations": int(phase.num_generations),
        "max_completion_length": int(phase.max_completion_length),
        "temperature": float(phase.temperature),
        "top_p": float(phase.top_p),
        "beta": float(phase.beta),
        "epsilon": float(phase.epsilon),
        "num_iterations": int(phase.num_iterations),
        "loss_type": str(phase.loss_type),
        "scale_rewards": str(phase.scale_rewards),
        "logging_steps": int(phase.logging_steps),
        "save_steps": int(phase.save_steps),
        "remove_unused_columns": False,
        "use_vllm": bool(phase.use_vllm),
        "vllm_mode": str(phase.vllm_mode),
        "report_to": [],
    }

    signature = inspect.signature(grpo_config_cls.__init__)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return grpo_config_cls(**filtered)


def _build_lora_config(lora: LoraTuningConfig) -> Any:
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError(
            "LoRA tuning selected, but PEFT is not installed. "
            "Install train dependencies first: python -m pip install -e .[train]"
        ) from exc

    task_type_token = str(lora.task_type or "CAUSAL_LM").strip().upper()
    task_type = getattr(TaskType, task_type_token, TaskType.CAUSAL_LM)
    return LoraConfig(
        r=max(1, int(lora.r)),
        lora_alpha=max(1, int(lora.alpha)),
        lora_dropout=float(lora.dropout),
        target_modules=list(lora.target_modules),
        bias=str(lora.bias),
        task_type=task_type,
    )



def _train_grpo_phase(
    model_name_or_path: str,
    phase: KimiGRPOPhaseConfig,
    rows: list[dict[str, Any]],
    reward_function: Any,
    output_dir: Path,
    tuning_mode: str,
    lora: LoraTuningConfig,
) -> dict[str, Any]:
    Dataset, GRPOConfig, GRPOTrainer = _require_training_stack()

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows)
    args = _safe_build_grpo_config(phase=phase, output_dir=str(output_dir), grpo_config_cls=GRPOConfig)

    trainer_kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "args": args,
        "reward_funcs": reward_function,
        "train_dataset": dataset,
    }

    if str(tuning_mode).strip().lower() == "lora":
        trainer_signature = inspect.signature(GRPOTrainer.__init__)
        if "peft_config" not in trainer_signature.parameters:
            raise RuntimeError("Installed TRL version does not expose peft_config in GRPOTrainer.")
        trainer_kwargs["peft_config"] = _build_lora_config(lora)

    trainer = GRPOTrainer(**trainer_kwargs)
    train_output = trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))

    global_step = int(getattr(train_output, "global_step", 0))
    training_loss = float(getattr(train_output, "training_loss", 0.0))

    return {
        "model_path": str(final_dir),
        "global_step": global_step,
        "training_loss": training_loss,
        "train_rows": len(rows),
        "tuning_mode": str(tuning_mode).strip().lower() or "full",
    }


def _resolve_initial_models(cfg: SelfPlayTrainingConfig) -> tuple[str, str]:
    topology = str(cfg.model_topology).strip().lower()
    if topology == "shared":
        shared = str(cfg.shared_model_name_or_path).strip()
        if not shared:
            shared = str(cfg.answerer_phase.model_name_or_path).strip() or str(cfg.generator_phase.model_name_or_path).strip()
        return shared, shared
    return str(cfg.generator_phase.model_name_or_path), str(cfg.answerer_phase.model_name_or_path)



def _fallback_generated_tasks(
    base_tasks: list[TaskInstance],
    round_index: int,
    count: int,
    rng: random.Random,
) -> list[TaskInstance]:
    if not base_tasks:
        return []

    selected = list(base_tasks)
    rng.shuffle(selected)
    selected = selected[: max(1, count)]

    out: list[TaskInstance] = []
    for idx, task in enumerate(selected):
        metadata = dict(task.metadata or {})
        metadata.update(
            {
                "generated_by": "fallback_generator",
                "difficulty": "hard",
                "round": round_index,
                "scenario": "adversarial_trace",
                "grader": {
                    "type": "difficulty_exact_match",
                    "answer_type": "node_id",
                    "case_sensitive": True,
                    "reward_profile": "hard",
                },
            }
        )
        out.append(
            TaskInstance(
                task_id=f"adv_r{round_index}_{idx}",
                task_type="adversarial_trace",
                question=f"[Adversarial] {task.question}",
                answer=task.answer,
                supporting_edges=list(task.supporting_edges),
                metadata=metadata,
            )
        )
    return out



def _sample_generated_tasks_with_model(
    model_name_or_path: str,
    prompts: list[str],
    round_index: int,
    count: int,
    max_support_edges: int,
) -> list[TaskInstance]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if count <= 0:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    import torch

    device = next(model.parameters()).device
    generated: list[TaskInstance] = []

    for prompt in prompts:
        if len(generated) >= count:
            break
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_ids = output[0][encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        candidate = parse_generated_task_completion(completion, max_support_edges=max_support_edges)
        if not candidate.is_valid:
            continue

        metadata = {
            "generated_by": "generator_model",
            "round": round_index,
            "difficulty": "hard",
            "scenario": "adversarial_trace",
            "grader": {
                "type": "difficulty_exact_match",
                "answer_type": "node_id",
                "case_sensitive": True,
                "reward_profile": "hard",
            },
        }
        generated.append(
            TaskInstance(
                task_id=f"adv_r{round_index}_{len(generated)}",
                task_type=candidate.task_type,
                question=candidate.question,
                answer=candidate.answer,
                supporting_edges=list(candidate.supporting_edges),
                metadata=metadata,
            )
        )

    return generated



def _select_answerer_tasks(
    seed_tasks: list[TaskInstance],
    generated_tasks: list[TaskInstance],
    cfg: SelfPlayTrainingConfig,
    rng: random.Random,
) -> list[TaskInstance]:
    seed_pick = list(seed_tasks)
    gen_pick = list(generated_tasks)
    rng.shuffle(seed_pick)
    rng.shuffle(gen_pick)

    chosen = seed_pick[: max(1, cfg.seed_tasks_per_round)]
    chosen.extend(gen_pick[: max(1, cfg.generated_tasks_per_round)])
    return chosen



def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")



def _save_tasks(path: Path, tasks: list[TaskInstance]) -> None:
    payload = []
    for task in tasks:
        payload.append(
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question": task.question,
                "answer": task.answer,
                "supporting_edges": [
                    {
                        "src": edge.src,
                        "rel": edge.rel,
                        "dst": edge.dst,
                        "confidence": float(edge.confidence),
                    }
                    for edge in task.supporting_edges
                ],
                "metadata": dict(task.metadata or {}),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")



def run_adversarial_self_play(
    env_config: EnvironmentConfig,
    training_config: SelfPlayTrainingConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    effective_dry_run = bool(dry_run or training_config.dry_run)
    topology = str(training_config.model_topology).strip().lower() or "dual"
    phase_schedule = str(training_config.phase_schedule).strip().lower() or "generator_answerer"
    tuning_mode = str(training_config.tuning_mode).strip().lower() or "full"

    run_dir = Path(training_config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = OSINTEnvironment(env_config, llm=build_llm_client(env_config.llm))
    seed_tasks = list(env.tasks)

    generator_model, answerer_model = _resolve_initial_models(training_config)

    rng = random.Random(env_config.seed)
    rounds_payload: list[dict[str, Any]] = []
    rolling_generated_tasks = _fallback_generated_tasks(
        base_tasks=seed_tasks,
        round_index=0,
        count=training_config.generated_tasks_per_round,
        rng=rng,
    )
    if not rolling_generated_tasks:
        rolling_generated_tasks = list(seed_tasks[: max(1, training_config.generated_tasks_per_round)])

    for round_index in range(1, max(1, training_config.rounds) + 1):
        round_dir = run_dir / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        answerer_pre_tasks: list[TaskInstance] = []
        answerer_pre_dataset_path: Path | None = None
        answerer_pre_train_result: dict[str, Any] | None = None

        if phase_schedule == "answerer_generator_answerer":
            answerer_pre_tasks = _select_answerer_tasks(
                seed_tasks=seed_tasks,
                generated_tasks=rolling_generated_tasks,
                cfg=training_config,
                rng=rng,
            )
            answerer_pre_rows = _build_answerer_rows(answerer_pre_tasks)
            answerer_pre_dataset_path = round_dir / "answerer_pre_dataset.json"
            _save_rows(answerer_pre_dataset_path, answerer_pre_rows)

            answerer_pre_train_result = {
                "model_path": answerer_model,
                "global_step": 0,
                "training_loss": 0.0,
                "train_rows": len(answerer_pre_rows),
                "skipped": effective_dry_run,
                "tuning_mode": tuning_mode,
            }

            if not effective_dry_run:
                answerer_pre_reward = AnswererRewardFunction(graph=env.graph)
                answerer_pre_train_result = _train_grpo_phase(
                    model_name_or_path=answerer_model,
                    phase=training_config.answerer_phase,
                    rows=answerer_pre_rows,
                    reward_function=answerer_pre_reward,
                    output_dir=round_dir / f"{training_config.answerer_phase.output_subdir}_pre",
                    tuning_mode=tuning_mode,
                    lora=training_config.lora,
                )
                answerer_model = str(answerer_pre_train_result["model_path"])
                if topology == "shared":
                    generator_model = answerer_model

        generator_rows = _build_generator_rows(env=env, cfg=training_config, rng=rng)
        generator_dataset_path = round_dir / "generator_dataset.json"
        _save_rows(generator_dataset_path, generator_rows)

        generator_train_result: dict[str, Any] = {
            "model_path": generator_model,
            "global_step": 0,
            "training_loss": 0.0,
            "train_rows": len(generator_rows),
            "skipped": effective_dry_run,
            "tuning_mode": tuning_mode,
        }

        if not effective_dry_run:
            generator_reward = GeneratorRewardFunction(
                graph=env.graph,
                answerer_judge=AnswererJudge(
                    model_name_or_path=answerer_model,
                    max_new_tokens=training_config.answerer_judge_max_new_tokens,
                ),
                weights=training_config.generator_reward_weights,
                max_support_edges=training_config.max_support_edges,
            )
            generator_train_result = _train_grpo_phase(
                model_name_or_path=generator_model,
                phase=training_config.generator_phase,
                rows=generator_rows,
                reward_function=generator_reward,
                output_dir=round_dir / training_config.generator_phase.output_subdir,
                tuning_mode=tuning_mode,
                lora=training_config.lora,
            )
            generator_model = str(generator_train_result["model_path"])
            if topology == "shared":
                answerer_model = generator_model

        generated_tasks: list[TaskInstance]
        if effective_dry_run:
            generated_tasks = _fallback_generated_tasks(
                base_tasks=seed_tasks,
                round_index=round_index,
                count=training_config.generated_tasks_per_round,
                rng=rng,
            )
        else:
            generated_tasks = _sample_generated_tasks_with_model(
                model_name_or_path=generator_model,
                prompts=[row["prompt"] for row in generator_rows],
                round_index=round_index,
                count=training_config.generated_tasks_per_round,
                max_support_edges=training_config.max_support_edges,
            )
            if not generated_tasks:
                generated_tasks = _fallback_generated_tasks(
                    base_tasks=seed_tasks,
                    round_index=round_index,
                    count=training_config.generated_tasks_per_round,
                    rng=rng,
                )

        if generated_tasks:
            rolling_generated_tasks = list(generated_tasks)

        generated_tasks_path = round_dir / "generated_tasks.json"
        _save_tasks(generated_tasks_path, generated_tasks)

        answerer_tasks = _select_answerer_tasks(
            seed_tasks=seed_tasks,
            generated_tasks=generated_tasks,
            cfg=training_config,
            rng=rng,
        )
        answerer_rows = _build_answerer_rows(answerer_tasks)
        answerer_dataset_path = round_dir / "answerer_dataset.json"
        _save_rows(answerer_dataset_path, answerer_rows)

        answerer_train_result: dict[str, Any] = {
            "model_path": answerer_model,
            "global_step": 0,
            "training_loss": 0.0,
            "train_rows": len(answerer_rows),
            "skipped": effective_dry_run,
            "tuning_mode": tuning_mode,
        }

        if not effective_dry_run:
            answerer_reward = AnswererRewardFunction(graph=env.graph)
            answerer_train_result = _train_grpo_phase(
                model_name_or_path=answerer_model,
                phase=training_config.answerer_phase,
                rows=answerer_rows,
                reward_function=answerer_reward,
                output_dir=round_dir / training_config.answerer_phase.output_subdir,
                tuning_mode=tuning_mode,
                lora=training_config.lora,
            )
            answerer_model = str(answerer_train_result["model_path"])
            if topology == "shared":
                generator_model = answerer_model

        artifacts = _RoundArtifacts(
            round_index=round_index,
            generator_dataset_path=str(generator_dataset_path),
            answerer_dataset_path=str(answerer_dataset_path),
            generated_tasks_path=str(generated_tasks_path),
        )

        rounds_payload.append(
            {
                "round": round_index,
                "dry_run": effective_dry_run,
                "phase_schedule": phase_schedule,
                "generator": generator_train_result,
                "answerer": answerer_train_result,
                "answerer_pre": answerer_pre_train_result,
                "generated_task_count": len(generated_tasks),
                "answerer_task_count": len(answerer_tasks),
                "answerer_pre_task_count": len(answerer_pre_tasks),
                "artifacts": {
                    "generator_dataset": artifacts.generator_dataset_path,
                    "answerer_dataset": artifacts.answerer_dataset_path,
                    "generated_tasks": artifacts.generated_tasks_path,
                    "answerer_pre_dataset": str(answerer_pre_dataset_path) if answerer_pre_dataset_path else "",
                },
            }
        )

    final_payload = {
        "dry_run": effective_dry_run,
        "output_dir": str(run_dir),
        "model_topology": topology,
        "phase_schedule": phase_schedule,
        "tuning_mode": tuning_mode,
        "rounds": rounds_payload,
        "final_models": {
            "generator": generator_model,
            "answerer": answerer_model,
        },
        "kimi_objective_mapping": {
            "grouped_rollouts": "TRL GRPO num_generations",
            "mean_centered_advantage": "GRPO relative reward baseline",
            "token_level_clipping": "GRPO epsilon clipping over policy ratios",
            "reference_regularization": "GRPO beta KL term",
            "toggle_self_play": "Alternating generator and answerer rounds",
        },
    }

    summary_path = run_dir / "self_play_summary.json"
    summary_path.write_text(json.dumps(final_payload, indent=2, sort_keys=True), encoding="utf-8")
    final_payload["summary_path"] = str(summary_path)

    return final_payload
