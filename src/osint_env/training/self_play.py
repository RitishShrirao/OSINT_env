from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from osint_env.data.generator import (
    build_swarm_v2_canonical_subgraph,
    build_swarm_v2_path_candidates,
    build_swarm_v2_tool_trace,
    emit_swarm_v2_question,
    select_swarm_v2_answer,
    trace_swarm_v2_path,
)
from osint_env.domain.models import Edge, EnvironmentConfig, TaskInstance
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.reward import compute_graph_f1
from osint_env.llm import build_llm_client
from osint_env.training.config import (
    KimiGRPOPhaseConfig,
    LoraTuningConfig,
    SelfPlayTrainingConfig,
    SwarmV2SwarmConfig,
)
from osint_env.training.rewards import (
    AnswererJudge,
    AnswererRewardFunction,
    GeneratorRewardFunction,
    SwarmV2ReplayValidator,
    decode_completion_text,
    extract_answer_from_completion,
    normalize_answer,
    parse_generated_task_completion,
)


@dataclass(slots=True)
class _RoundArtifacts:
    round_index: int
    generator_dataset_path: str
    answerer_dataset_path: str
    generated_tasks_path: str


def _is_true_env(value: str | None) -> bool:
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "y", "on"}


def _resolve_hf_upload_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = str(os.getenv(env_name, "")).strip()
        if token:
            return token
    return ""


def _slugify_hf_repo_name(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip().lower())
    token = re.sub(r"-{2,}", "-", token).strip("-.")
    return token


def _default_hf_checkpoint_repo_id(run_dir: Path) -> str:
    explicit = str(os.getenv("OSINT_HF_CHECKPOINT_REPO_ID", "")).strip()
    if explicit:
        return explicit

    space_id = str(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or "").strip()
    if "/" not in space_id:
        return ""

    owner, _, space_name = space_id.partition("/")
    suffix = str(os.getenv("OSINT_HF_CHECKPOINT_REPO_SUFFIX", "-checkpoints")).strip() or "-checkpoints"
    repo_name = _slugify_hf_repo_name(f"{space_name}{suffix}") or "osint-self-play-checkpoints"
    return f"{owner}/{repo_name}"


def _hf_checkpoint_repo_prefix(run_dir: Path) -> str:
    explicit = str(os.getenv("OSINT_HF_CHECKPOINT_PATH_PREFIX", "")).strip().strip("/")
    if explicit:
        return explicit
    return _slugify_hf_repo_name(run_dir.name) or "self-play"


def _hf_relative_repo_path(local_path: Path, run_dir: Path) -> str:
    prefix = _hf_checkpoint_repo_prefix(run_dir)
    try:
        relative = local_path.relative_to(run_dir).as_posix()
    except ValueError:
        relative = local_path.name
    return f"{prefix}/{relative}".strip("/")


def _maybe_upload_folder_to_hf(local_dir: Path, run_dir: Path, commit_message: str) -> None:
    repo_id = _default_hf_checkpoint_repo_id(run_dir)
    token = _resolve_hf_upload_token()
    if not repo_id or not token or not local_dir.exists():
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[self_play][hf_upload] huggingface_hub missing; skipping checkpoint upload.")
        return

    repo_type = str(os.getenv("OSINT_HF_CHECKPOINT_REPO_TYPE", "model")).strip() or "model"
    private = _is_true_env(os.getenv("OSINT_HF_CHECKPOINT_REPO_PRIVATE", "1"))
    path_in_repo = _hf_relative_repo_path(local_dir, run_dir)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"],
    )
    print(f"[self_play][hf_upload] uploaded {local_dir} -> {repo_type}:{repo_id}/{path_in_repo}")


def _maybe_upload_file_to_hf(local_file: Path, run_dir: Path, commit_message: str) -> None:
    repo_id = _default_hf_checkpoint_repo_id(run_dir)
    token = _resolve_hf_upload_token()
    if not repo_id or not token or not local_file.exists():
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[self_play][hf_upload] huggingface_hub missing; skipping artifact upload.")
        return

    repo_type = str(os.getenv("OSINT_HF_CHECKPOINT_REPO_TYPE", "model")).strip() or "model"
    private = _is_true_env(os.getenv("OSINT_HF_CHECKPOINT_REPO_PRIVATE", "1"))
    path_in_repo = _hf_relative_repo_path(local_file, run_dir)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_file),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    print(f"[self_play][hf_upload] uploaded {local_file} -> {repo_type}:{repo_id}/{path_in_repo}")



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


def _edge_payload(edge: Edge) -> dict[str, Any]:
    return {
        "src": edge.src,
        "rel": edge.rel,
        "dst": edge.dst,
        "confidence": float(edge.confidence),
    }


def _edges_from_payload(rows: Any, max_edges: int) -> list[Edge]:
    if not isinstance(rows, list):
        return []
    edges: list[Edge] = []
    for row in rows[:max_edges]:
        if not isinstance(row, dict):
            continue
        src = str(row.get("src", "")).strip()
        rel = str(row.get("rel", "")).strip()
        dst = str(row.get("dst", "")).strip()
        if not src or not rel or not dst:
            continue
        try:
            confidence = float(row.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        edges.append(Edge(src=src, rel=rel, dst=dst, confidence=confidence))
    return edges


def _compact_shared_context(
    shared_context: dict[str, Any],
    max_nodes: int = 8,
    max_edges: int = 6,
) -> dict[str, Any]:
    return {
        "nodes": list(shared_context.get("nodes", []))[:max_nodes],
        "edges": list(shared_context.get("edges", []))[:max_edges],
    }


def _task_shared_context(
    env: OSINTEnvironment,
    task: TaskInstance,
    cfg: SelfPlayTrainingConfig,
) -> dict[str, Any]:
    metadata = dict(task.metadata or {})
    canonical_graph = metadata.get("canonical_graph")
    if isinstance(canonical_graph, dict):
        return {
            "nodes": list(canonical_graph.get("nodes", []))[: cfg.swarm_v2.shared_context.max_nodes],
            "edges": list(canonical_graph.get("edges", []))[: cfg.swarm_v2.shared_context.max_edges],
        }

    deterministic_seed = sum(ord(ch) for ch in task.task_id)
    return _graph_context_for_prompt(
        env=env,
        max_nodes=cfg.swarm_v2.shared_context.max_nodes,
        max_edges=cfg.swarm_v2.shared_context.max_edges,
        rng=random.Random(deterministic_seed),
    )


def _swarm_v2_worker_packets(
    canonical_candidate: dict[str, Any],
    shared_context: dict[str, Any],
    swarm_cfg: SwarmV2SwarmConfig,
) -> dict[str, Any]:
    path_edges = _edges_from_payload(
        canonical_candidate.get("path", canonical_candidate.get("edges", [])),
        max_edges=max(1, swarm_cfg.max_depth * 2),
    )
    if not path_edges:
        path_edges = _edges_from_payload(canonical_candidate.get("edges", []), max_edges=2)
    relation_path = [edge.rel for edge in path_edges]
    start_node = path_edges[0].src if path_edges else ""
    return {
        "path_agent": {
            "path_edges": [_edge_payload(edge) for edge in path_edges],
            "goal": "Choose one contiguous replayable path from the canonical candidate.",
        },
        "question_agent": {
            "start_node": start_node,
            "relation_path": relation_path,
            "goal": "Write a compact question that describes the path without leaking the answer.",
        },
        "context_agent": {
            "shared_context": _compact_shared_context(shared_context),
            "goal": "Keep support/context usage compact and graph-grounded.",
        },
        "planner": {
            "max_agents": int(swarm_cfg.max_agents),
            "max_breadth": int(swarm_cfg.max_breadth),
            "max_depth": int(swarm_cfg.max_depth),
        },
    }


def _serialize_tool_trace(tool_trace: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for call in tool_trace or []:
        tool_name = getattr(call, "tool_name", "")
        args = getattr(call, "args", {})
        output = getattr(call, "output", {})
        if not tool_name:
            continue
        serialized.append(
            {
                "tool_name": str(tool_name),
                "args": dict(args) if isinstance(args, dict) else {},
                "output": dict(output) if isinstance(output, dict) else {},
            }
        )
    return serialized



def _canonical_example_payload(
    graph: Any,
    canonical_candidate: dict[str, Any],
    swarm_cfg: SwarmV2SwarmConfig,
) -> dict[str, Any]:
    candidate_edges = _edges_from_payload(canonical_candidate.get("edges", []), max_edges=2)
    traced_edges = trace_swarm_v2_path(graph, candidate_edges) or candidate_edges
    if not traced_edges:
        return {
            "question": "Which entity is reached by following the provided replayable relation path?",
            "answer": "",
            "task_type": "swarm_v2_trace",
            "supporting_edges": [],
            "subagent_outputs": ["path_agent: no replayable edge"],
            "orchestrator": {
                "spawn_count": 1,
                "finished_subtasks": 1,
                "critical_steps": 1,
                "breadth": 1,
                "depth": 1,
            },
        }

    traced_edges = traced_edges[:2]
    spawn_count = min(swarm_cfg.max_agents, max(1, len(traced_edges) + 1))
    return {
        "question": emit_swarm_v2_question(traced_edges),
        "answer": select_swarm_v2_answer(traced_edges),
        "task_type": f"swarm_v2_{len(traced_edges)}hop_trace",
        "supporting_edges": [_edge_payload(edge) for edge in traced_edges],
        "subagent_outputs": [
            f"path_agent_{idx}: {edge.src} --{edge.rel}--> {edge.dst}"
            for idx, edge in enumerate(traced_edges)
        ]
        + [
            "question_agent: emitted compact relation-path question",
            "context_agent: kept shared context focused on replayable edges",
        ],
        "orchestrator": {
            "spawn_count": spawn_count,
            "finished_subtasks": spawn_count,
            "critical_steps": max(1, len(traced_edges)),
            "breadth": min(swarm_cfg.max_breadth, spawn_count),
            "depth": min(swarm_cfg.max_depth, 1),
        },
    }


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
        "Return ONLY one compact JSON object. Do not use markdown. Do not add prose.\n"
        "Required schema: {\"answer\": \"<entity_or_value>\"}\n"
        "Valid example: {\"answer\": \"user_7\"}\n"
        f"Question: {question}"
    )


def _swarm_v2_answer_prompt(
    question: str,
    shared_context: dict[str, Any],
    swarm_cfg: SwarmV2SwarmConfig,
) -> str:
    del swarm_cfg  # kept for signature compatibility
    compact_context = _compact_shared_context(shared_context)
    return (
        "You answer one OSINT graph question using ONLY the shared context.\n"
        "Output rules:\n"
        "- Return ONLY one compact JSON object. No markdown. No prose. End with }.\n"
        "- Required keys: answer, supporting_edges, orchestrator.\n"
        "- supporting_edges: list of {src, rel, dst, confidence} taken from shared edges.\n"
        "- orchestrator integer keys: spawn_count, finished_subtasks, critical_steps, breadth, depth.\n"
        "Example schema:\n"
        "{\"answer\":\"user_7\",\"supporting_edges\":[{\"src\":\"alias_7_123\",\"rel\":\"alias_of\","
        "\"dst\":\"user_7\",\"confidence\":1.0}],\"orchestrator\":{\"spawn_count\":2,"
        "\"finished_subtasks\":2,\"critical_steps\":1,\"breadth\":2,\"depth\":1}}\n"
        f"Shared context: {json.dumps(compact_context, separators=(',', ':'), sort_keys=True)}\n"
        f"Question: {question}\n"
        "JSON:"
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


def _build_swarm_v2_answerer_rows(
    env: OSINTEnvironment,
    tasks: list[TaskInstance],
    cfg: SelfPlayTrainingConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        shared_context = _task_shared_context(env=env, task=task, cfg=cfg)

        rows.append(
            {
                "prompt": _swarm_v2_answer_prompt(
                    question=task.question,
                    shared_context=shared_context,
                    swarm_cfg=cfg.swarm_v2.answerer_swarm,
                ),
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


def _swarm_v2_generator_prompt(
    graph: Any,
    shared_context: dict[str, Any],
    canonical_candidate: dict[str, Any],
    anchor_questions: list[str],
    swarm_cfg: SwarmV2SwarmConfig,
    canonical_graph_mode: str,
) -> str:
    anchors = "\n".join(f"- {question}" for question in anchor_questions)
    canonical_mode = str(canonical_graph_mode).strip().lower() or "generate"
    example_payload = _canonical_example_payload(graph, canonical_candidate, swarm_cfg)
    worker_packets = _swarm_v2_worker_packets(
        canonical_candidate=canonical_candidate,
        shared_context=shared_context,
        swarm_cfg=swarm_cfg,
    )
    canonical_instruction = (
        "You may propose canonical_graph updates when they improve replayability and keep it graph-grounded."
        if canonical_mode == "generate"
        else "Reuse the provided canonical candidate as-is; do not add, remove, or modify canonical_graph nodes/edges."
    )
    canonical_compact = _compact_shared_context(canonical_candidate)
    return (
        "You coordinate a compact multi-agent OSINT task-generation swarm.\n"
        "Output rules:\n"
        "- Return ONLY one JSON object. No markdown. No prose. End with }.\n"
        "- Required keys: question, answer, task_type, supporting_edges, subagent_outputs, orchestrator.\n"
        "- Optional keys: canonical_graph, validation.\n"
        "- supporting_edges: non-empty list of {src, rel, dst, confidence}, taken from canonical edges.\n"
        "- supporting_edges must form one contiguous replayable path. Keep it compact.\n"
        "- Do NOT emit verbose tool traces or neighbor dumps; replay tools are derived from supporting_edges.\n"
        "- answer = final dst of the trace. question describes the path without leaking the answer.\n"
        "- subagent_outputs: 2-4 terse strings summarizing path_agent/question_agent/context_agent work.\n"
        "- orchestrator: integer keys spawn_count, finished_subtasks, critical_steps, breadth, depth.\n"
        f"- canonical_graph_mode={canonical_mode}: {canonical_instruction}\n"
        "- Favor minimal shared context per worker so question generation stays parallel-friendly.\n"
        "Example (copy schema, not values):\n"
        f"{json.dumps(example_payload, separators=(',', ':'), sort_keys=True)}\n"
        "Worker packets:\n"
        f"{json.dumps(worker_packets, separators=(',', ':'), sort_keys=True)}\n"
        "Canonical candidate (use these edges):\n"
        f"{json.dumps(canonical_compact, separators=(',', ':'), sort_keys=True)}\n"
        "Shared context:\n"
        f"{json.dumps(_compact_shared_context(shared_context), separators=(',', ':'), sort_keys=True)}\n"
        f"Avoid these prior questions: {anchors}\n"
        "JSON:"
    )


def _build_swarm_v2_generator_rows(
    env: OSINTEnvironment,
    cfg: SelfPlayTrainingConfig,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    canonical_candidates: list[dict[str, Any]] = []
    existing_questions = [task.question for task in env.tasks]
    path_candidates = build_swarm_v2_path_candidates(
        env.graph,
        rng=rng,
        count=max(1, cfg.generator_prompts_per_round),
        min_hops=2,
        max_hops=cfg.swarm_v2.validation.max_path_hops,
    )
    for idx, path_edges in enumerate(path_candidates):
        shared_context = _graph_context_for_prompt(
            env=env,
            max_nodes=cfg.swarm_v2.shared_context.max_nodes,
            max_edges=cfg.swarm_v2.shared_context.max_edges,
            rng=rng,
        )
        canonical_candidate = build_swarm_v2_canonical_subgraph(
            env.graph,
            path_edges=path_edges,
            max_extra_edges=max(0, cfg.swarm_v2.shared_context.max_edges - len(path_edges)),
        )
        anchor_sample_size = min(5, len(existing_questions))
        anchor_sample = rng.sample(existing_questions, k=anchor_sample_size) if anchor_sample_size > 0 else []
        prompt = _swarm_v2_generator_prompt(
            graph=env.graph,
            shared_context=shared_context,
            canonical_candidate=canonical_candidate,
            anchor_questions=anchor_sample,
            swarm_cfg=cfg.swarm_v2.generator_swarm,
            canonical_graph_mode=cfg.canonical_graph_mode,
        )
        rows.append(
            {
                "prompt": prompt,
                "candidate_id": f"candidate_{idx}",
                "canonical_graph_json": json.dumps(canonical_candidate, sort_keys=True),
                "shared_context_json": json.dumps(shared_context, sort_keys=True),
                "worker_packets_json": json.dumps(
                    _swarm_v2_worker_packets(
                        canonical_candidate=canonical_candidate,
                        shared_context=shared_context,
                        swarm_cfg=cfg.swarm_v2.generator_swarm,
                    ),
                    sort_keys=True,
                ),
            }
        )
        canonical_candidates.append(canonical_candidate)
    return rows, canonical_candidates



def _safe_build_grpo_config(
    phase: KimiGRPOPhaseConfig,
    output_dir: str,
    grpo_config_cls: Any,
    report_to: list[str] | None = None,
    run_name: str = "",
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
        "repetition_penalty": float(phase.repetition_penalty),
        "beta": float(phase.beta),
        "epsilon": float(phase.epsilon),
        "num_iterations": int(phase.num_iterations),
        "loss_type": str(phase.loss_type),
        "scale_rewards": str(phase.scale_rewards),
        "logging_steps": int(phase.logging_steps),
        "save_steps": int(phase.save_steps),
        "save_total_limit": int(phase.save_total_limit),
        "optim": str(phase.optim),
        "bf16": bool(phase.bf16),
        "tf32": bool(phase.tf32),
        "gradient_checkpointing": bool(phase.gradient_checkpointing),
        "dataloader_num_workers": int(phase.dataloader_num_workers),
        "dataloader_persistent_workers": bool(phase.dataloader_persistent_workers),
        "dataloader_prefetch_factor": int(phase.dataloader_prefetch_factor),
        "generation_batch_size": int(phase.generation_batch_size),
        "max_prompt_length": int(phase.max_prompt_length),
        "remove_unused_columns": False,
        "use_vllm": bool(phase.use_vllm),
        "vllm_mode": str(phase.vllm_mode),
        "report_to": list(report_to or []),
    }
    if str(run_name).strip():
        kwargs["run_name"] = str(run_name).strip()

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


def _coerce_named_reward_func(reward_function: Any) -> Any:
    """Return a callable with a stable __name__ for TRL compatibility."""
    if hasattr(reward_function, "__name__") and str(getattr(reward_function, "__name__", "")).strip():
        return reward_function

    # TRL versions that introspect reward_funcs[i].__name__ require this attribute.
    if callable(reward_function):
        name = reward_function.__class__.__name__ or "reward_func"
        try:
            setattr(reward_function, "__name__", name)
            return reward_function
        except Exception:
            def _wrapped_reward(*args: Any, **kwargs: Any) -> Any:
                return reward_function(*args, **kwargs)

            _wrapped_reward.__name__ = name
            return _wrapped_reward
    return reward_function



def _train_grpo_phase(
    model_name_or_path: str,
    phase: KimiGRPOPhaseConfig,
    rows: list[dict[str, Any]],
    reward_function: Any,
    output_dir: Path,
    tuning_mode: str,
    lora: LoraTuningConfig,
    report_to: list[str] | None = None,
    run_name: str = "",
) -> dict[str, Any]:
    Dataset, GRPOConfig, GRPOTrainer = _require_training_stack()

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows)
    args = _safe_build_grpo_config(
        phase=phase,
        output_dir=str(output_dir),
        grpo_config_cls=GRPOConfig,
        report_to=report_to,
        run_name=run_name,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "args": args,
        "reward_funcs": _coerce_named_reward_func(reward_function),
        "train_dataset": dataset,
    }

    if str(tuning_mode).strip().lower() == "lora":
        trainer_signature = inspect.signature(GRPOTrainer.__init__)
        if "peft_config" not in trainer_signature.parameters:
            raise RuntimeError("Installed TRL version does not expose peft_config in GRPOTrainer.")
        trainer_kwargs["peft_config"] = _build_lora_config(lora)

    phase_label = str(run_name).strip() or str(output_dir.name)
    print(f"[self_play] Starting phase: {phase_label} rows={len(rows)} max_steps={phase.max_steps}")
    strict_asserts = str(os.getenv("OSINT_TRAIN_STRICT_ASSERTS", "")).strip().lower() in {"1", "true", "yes", "on"}
    trainer = GRPOTrainer(**trainer_kwargs)
    tracked_params = [
        (name, param)
        for name, param in trainer.model.named_parameters()
        if getattr(param, "requires_grad", False)
    ][:32]
    pre_update_fingerprint = {
        name: float(param.detach().float().abs().mean().item())
        for name, param in tracked_params
    }
    train_output = trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    trainer_tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if trainer_tokenizer is not None and hasattr(trainer_tokenizer, "save_pretrained"):
        trainer_tokenizer.save_pretrained(str(final_dir))
    checkpoint_dirs = [str(path) for path in sorted(output_dir.glob("checkpoint-*")) if path.is_dir()]

    global_step = int(getattr(train_output, "global_step", 0))
    training_loss = float(getattr(train_output, "training_loss", 0.0))

    total_param_count = int(sum(param.numel() for param in trainer.model.parameters()))
    result = {
        "model_path": str(final_dir),
        "final_model_path": str(final_dir),
        "phase_output_dir": str(output_dir),
        "checkpoint_dirs": checkpoint_dirs,
        "global_step": global_step,
        "training_loss": training_loss,
        "train_rows": len(rows),
        "tuning_mode": str(tuning_mode).strip().lower() or "full",
        "is_full_finetune": str(tuning_mode).strip().lower() != "lora",
    }

    log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    reward_values = [float(row.get("reward")) for row in log_history if isinstance(row, dict) and "reward" in row]
    reward_std_values = [
        float(row.get("reward_std"))
        for row in log_history
        if isinstance(row, dict) and "reward_std" in row
    ]
    kl_values = [float(row.get("kl")) for row in log_history if isinstance(row, dict) and "kl" in row]
    grad_norm_values = [
        float(row.get("grad_norm"))
        for row in log_history
        if isinstance(row, dict) and "grad_norm" in row
    ]
    loss_values = [float(row.get("loss")) for row in log_history if isinstance(row, dict) and "loss" in row]
    entropy_values = [float(row.get("entropy")) for row in log_history if isinstance(row, dict) and "entropy" in row]

    trainable_params = [param for param in trainer.model.parameters() if getattr(param, "requires_grad", False)]
    grad_tensors = [param.grad for param in trainable_params if getattr(param, "grad", None) is not None]
    trainable_param_count = int(sum(param.numel() for param in trainable_params))
    params_with_grad = int(len(grad_tensors))
    nonzero_grad_tensors = int(
        sum(
            1
            for grad in grad_tensors
            if float(grad.detach().abs().sum().item()) > 0.0
        )
    )

    diagnostics = {
        "reward_min": min(reward_values) if reward_values else 0.0,
        "reward_max": max(reward_values) if reward_values else 0.0,
        "reward_std_max": max(reward_std_values) if reward_std_values else 0.0,
        "kl_max": max(kl_values) if kl_values else 0.0,
        "loss_abs_max": max((abs(value) for value in loss_values), default=0.0),
        "grad_norm_max": max(grad_norm_values) if grad_norm_values else 0.0,
        "entropy_min": min(entropy_values) if entropy_values else 0.0,
        "entropy_max": max(entropy_values) if entropy_values else 0.0,
        "total_param_count": total_param_count,
        "trainable_param_count": trainable_param_count,
        "trainable_fraction": (
            float(trainable_param_count / total_param_count)
            if total_param_count > 0
            else 0.0
        ),
        "params_with_grad": params_with_grad,
        "nonzero_grad_tensors": nonzero_grad_tensors,
        "fingerprint_param_count": len(pre_update_fingerprint),
        "fingerprint_changed_count": 0,
    }
    if pre_update_fingerprint:
        changed_count = 0
        for name, param in tracked_params:
            after_value = float(param.detach().float().abs().mean().item())
            before_value = pre_update_fingerprint.get(name, after_value)
            if abs(after_value - before_value) > 1e-9:
                changed_count += 1
        diagnostics["fingerprint_changed_count"] = changed_count
    result["diagnostics"] = diagnostics

    print(
        "[self_play][diagnostics] "
        f"{phase_label} reward_range=({diagnostics['reward_min']:.4f},{diagnostics['reward_max']:.4f}) "
        f"reward_std_max={diagnostics['reward_std_max']:.6f} "
        f"kl_max={diagnostics['kl_max']:.6f} "
        f"loss_abs_max={diagnostics['loss_abs_max']:.6f} "
        f"grad_norm_max={diagnostics['grad_norm_max']:.6f} "
        f"nonzero_grad_tensors={diagnostics['nonzero_grad_tensors']}/{max(1, diagnostics['params_with_grad'])} "
        f"fingerprint_changed={diagnostics['fingerprint_changed_count']}/{max(1, diagnostics['fingerprint_param_count'])}"
    )

    if strict_asserts:
        assert diagnostics["reward_max"] != diagnostics["reward_min"], (
            f"Constant reward detected in {phase_label}: {diagnostics['reward_min']}"
        )
        assert diagnostics["reward_std_max"] > 0.0, f"reward_std stayed zero in {phase_label}"
        assert diagnostics["kl_max"] > 0.0, f"KL stayed zero in {phase_label}"
        assert diagnostics["loss_abs_max"] > 0.0, f"Loss stayed zero in {phase_label}"
        assert diagnostics["grad_norm_max"] > 0.0, f"Grad norm stayed zero in {phase_label}"
        assert diagnostics["nonzero_grad_tensors"] > 0, f"No non-zero grads in {phase_label}"
        assert diagnostics["fingerprint_changed_count"] > 0, f"No parameter fingerprint change in {phase_label}"

    reward_debug = getattr(reward_function, "_debug_last_batch", None)
    if isinstance(reward_debug, dict):
        print(f"[reward_debug][last_batch] {phase_label} {json.dumps(reward_debug, sort_keys=True)}")

    print(
        "[self_play] Finished phase: "
        f"{phase_label} global_step={global_step} training_loss={training_loss} output={final_dir}"
    )
    return result


def _resolve_reporting(training_config: SelfPlayTrainingConfig, phase_name: str, round_index: int) -> tuple[list[str], str]:
    if not training_config.wandb_enabled:
        return [], ""
    if training_config.wandb_project:
        os.environ["WANDB_PROJECT"] = str(training_config.wandb_project)
    if training_config.wandb_entity:
        os.environ["WANDB_ENTITY"] = str(training_config.wandb_entity)
    prefix = str(training_config.wandb_run_name_prefix).strip() or "self-play"
    run_name = f"{prefix}-r{round_index:03d}-{phase_name}"
    return ["wandb"], run_name


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
    max_new_tokens: int,
) -> list[TaskInstance]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if count <= 0:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()

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
                max_new_tokens=max(64, int(max_new_tokens)),
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


def _save_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _generate_answerer_completion_texts_with_model(
    model_name_or_path: str,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device

    completions: list[str] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max(16, int(max_new_tokens)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output[0][encoded["input_ids"].shape[1] :]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return completions


def _top_validation_reasons(validation_reports: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for report in validation_reports:
        validation = report.get("validation", {}) if isinstance(report, dict) else {}
        reasons = validation.get("reasons", []) if isinstance(validation, dict) else []
        for reason in reasons:
            token = str(reason).strip()
            if not token:
                continue
            counts[token] = counts.get(token, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _run_post_training_evaluation(
    env_config: EnvironmentConfig,
    training_config: SelfPlayTrainingConfig,
    generator_model: str,
    answerer_models: dict[str, str],
    output_dir: Path,
    pipeline_mode: str,
    effective_dry_run: bool,
) -> dict[str, Any]:
    tasks_path = output_dir / "post_training_eval_generated_tasks.json"
    validation_path = output_dir / "post_training_eval_validation_reports.json"
    payload_path = output_dir / "post_training_evaluation.json"
    payload: dict[str, Any] = {
        "pipeline_mode": pipeline_mode,
        "generator_model": generator_model,
        "answerer_models": dict(answerer_models),
        "generated_tasks_path": str(tasks_path),
        "validation_reports_path": str(validation_path),
        "skipped": False,
    }

    if effective_dry_run:
        payload.update({"skipped": True, "reason": "dry_run"})
        _save_payload(validation_path, [])
        _save_payload(tasks_path, [])
        _save_payload(payload_path, payload)
        payload["path"] = str(payload_path)
        return payload

    try:
        env = OSINTEnvironment(env_config, llm=build_llm_client(env_config.llm))
        rng = random.Random(env_config.seed + 9973)
        validation_reports: list[dict[str, Any]] = []

        if pipeline_mode == "swarm_v2":
            generator_rows, prompt_canonical_candidates = _build_swarm_v2_generator_rows(env, training_config, rng)
            completion_texts = _sample_swarm_v2_completion_texts_with_model(
                env=env,
                cfg=training_config,
                model_name_or_path=generator_model,
                prompts=[row["prompt"] for row in generator_rows],
                count=max(1, training_config.post_training_eval_questions * 2),
                seen_questions=[task.question for task in env.tasks],
            )
            generated_tasks, validation_reports, _, _ = _materialize_swarm_v2_completions(
                env=env,
                cfg=training_config,
                completion_texts=completion_texts,
                round_index=max(1, training_config.rounds) + 1,
                seen_questions=[task.question for task in env.tasks],
                prompt_canonical_candidates=prompt_canonical_candidates,
            )
            if not generated_tasks:
                generated_tasks, validation_reports, _, _ = _materialize_swarm_v2_completions(
                    env=env,
                    cfg=training_config,
                    completion_texts=_fallback_swarm_v2_completion_texts(
                        env=env,
                        cfg=training_config,
                        round_index=max(1, training_config.rounds) + 1,
                        rng=rng,
                    ),
                    round_index=max(1, training_config.rounds) + 1,
                    seen_questions=[task.question for task in env.tasks],
                    prompt_canonical_candidates=None,
                )
            generated_tasks = generated_tasks[: max(1, training_config.post_training_eval_questions)]
            answer_rows = _build_swarm_v2_answerer_rows(env, generated_tasks, training_config)
            reward_fn = AnswererRewardFunction(
                graph=env.graph,
                pipeline_mode="swarm_v2",
                parl_max_parallel_hint=training_config.swarm_v2.answerer_swarm.max_agents,
            )
        else:
            generator_rows = _build_generator_rows(env=env, cfg=training_config, rng=rng)
            generated_tasks = _sample_generated_tasks_with_model(
                model_name_or_path=generator_model,
                prompts=[row["prompt"] for row in generator_rows],
                round_index=max(1, training_config.rounds) + 1,
                count=max(1, training_config.post_training_eval_questions),
                max_support_edges=training_config.max_support_edges,
                max_new_tokens=training_config.generated_task_max_new_tokens,
            )
            if not generated_tasks:
                generated_tasks = _fallback_generated_tasks(
                    base_tasks=list(env.tasks),
                    round_index=max(1, training_config.rounds) + 1,
                    count=max(1, training_config.post_training_eval_questions),
                    rng=rng,
                )
            answer_rows = _build_answerer_rows(generated_tasks)
            reward_fn = AnswererRewardFunction(graph=env.graph)

        _save_tasks(tasks_path, generated_tasks)
        _save_payload(validation_path, validation_reports)

        model_evaluations: dict[str, dict[str, Any]] = {}
        for model_label, answerer_model in answerer_models.items():
            answerer_completions = _generate_answerer_completion_texts_with_model(
                model_name_or_path=answerer_model,
                prompts=[row["prompt"] for row in answer_rows],
                max_new_tokens=training_config.post_training_eval_answer_max_new_tokens,
            )
            rewards = reward_fn(
                prompts=[row["prompt"] for row in answer_rows],
                completions=answerer_completions,
                answer=[row["answer"] for row in answer_rows],
                question=[row["question"] for row in answer_rows],
                supporting_edges_json=[row["supporting_edges_json"] for row in answer_rows],
                difficulty=[row["difficulty"] for row in answer_rows],
            )

            episodes: list[dict[str, Any]] = []
            for task, row, completion_text, reward in zip(generated_tasks, answer_rows, answerer_completions, rewards):
                support_edges = AnswererRewardFunction._parse_support_edges(row["supporting_edges_json"])
                pred_edges = AnswererRewardFunction._extract_predicted_edges(completion_text, support_edges)
                predicted_answer = normalize_answer(extract_answer_from_completion(completion_text))
                target_answer = normalize_answer(task.answer)
                graph_f1 = compute_graph_f1(pred_edges, support_edges)
                episodes.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "question": task.question,
                        "task_answer": target_answer,
                        "agent_answer": predicted_answer,
                        "reward": float(reward),
                        "graph_f1": float(graph_f1),
                        "success": int(predicted_answer == target_answer),
                        "support_edge_count": len(support_edges),
                        "predicted_edge_count": len(pred_edges),
                        "completion_length": len(completion_text),
                    }
                )

            episode_count = len(episodes)
            model_evaluations[model_label] = {
                "model_path": answerer_model,
                "episodes": episodes,
                "summary": {
                    "episodes": episode_count,
                    "task_success_rate": (
                        float(sum(row["success"] for row in episodes) / max(1, episode_count))
                        if episodes
                        else 0.0
                    ),
                    "avg_reward": (
                        float(sum(float(row["reward"]) for row in episodes) / max(1, episode_count))
                        if episodes
                        else 0.0
                    ),
                    "avg_graph_f1": (
                        float(sum(float(row["graph_f1"]) for row in episodes) / max(1, episode_count))
                        if episodes
                        else 0.0
                    ),
                    "avg_completion_length": (
                        float(sum(int(row["completion_length"]) for row in episodes) / max(1, episode_count))
                        if episodes
                        else 0.0
                    ),
                },
            }

        final_summary = model_evaluations.get("finetuned_answerer", {}).get("summary", {})
        baseline_summary = model_evaluations.get("original_answerer", {}).get("summary", {})
        summary = {
            "generated_task_count": len(generated_tasks),
            "generator_valid_rate": (
                float(len(generated_tasks) / max(1, len(validation_reports)))
                if validation_reports
                else 1.0
            ),
            "compared_models": sorted(model_evaluations.keys()),
            "finetuned_answerer": dict(final_summary),
            "original_answerer": dict(baseline_summary),
            "delta_vs_original": {
                "task_success_rate": float(final_summary.get("task_success_rate", 0.0) - baseline_summary.get("task_success_rate", 0.0)),
                "avg_reward": float(final_summary.get("avg_reward", 0.0) - baseline_summary.get("avg_reward", 0.0)),
                "avg_graph_f1": float(final_summary.get("avg_graph_f1", 0.0) - baseline_summary.get("avg_graph_f1", 0.0)),
            },
            "top_generator_invalid_reasons": _top_validation_reasons(validation_reports)[:5],
        }
        payload.update(
            {
                "summary": summary,
                "model_evaluations": model_evaluations,
            }
        )
    except Exception as exc:
        payload.update({"skipped": True, "reason": f"{type(exc).__name__}: {exc}"})

    if not tasks_path.exists():
        _save_payload(tasks_path, [])
    if not validation_path.exists():
        _save_payload(validation_path, [])
    _save_payload(payload_path, payload)
    payload["path"] = str(payload_path)
    return payload


def _fallback_swarm_v2_completion_texts(
    env: OSINTEnvironment,
    cfg: SelfPlayTrainingConfig,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    completion_texts: list[str] = []
    path_candidates = build_swarm_v2_path_candidates(
        env.graph,
        rng=rng,
        count=max(1, cfg.generated_tasks_per_round * 2),
        min_hops=2,
        max_hops=cfg.swarm_v2.validation.max_path_hops,
    )
    for idx, path_edges in enumerate(path_candidates):
        traced_edges = trace_swarm_v2_path(env.graph, path_edges)
        if not traced_edges:
            continue
        question = emit_swarm_v2_question(traced_edges)
        answer = select_swarm_v2_answer(traced_edges)
        canonical_graph = build_swarm_v2_canonical_subgraph(
            env.graph,
            path_edges=traced_edges,
            max_extra_edges=max(0, cfg.swarm_v2.shared_context.max_edges - len(traced_edges)),
        )
        spawn_count = min(
            cfg.swarm_v2.generator_swarm.max_agents,
            max(1, len(traced_edges) + 1),
        )
        payload = {
            "canonical_graph": canonical_graph,
            "question": question,
            "answer": answer,
            "task_type": f"swarm_v2_{len(traced_edges)}hop_trace",
            "supporting_edges": [
                {
                    "src": edge.src,
                    "rel": edge.rel,
                    "dst": edge.dst,
                    "confidence": float(edge.confidence),
                }
                for edge in traced_edges
            ],
            "tool_trace": build_swarm_v2_tool_trace(env.graph, traced_edges),
            "subagent_outputs": [
                f"path_agent_{edge_idx}: {edge.src} --{edge.rel}--> {edge.dst}"
                for edge_idx, edge in enumerate(traced_edges)
            ]
            + [
                f"question_agent: emitted deterministic relation-path question for round {round_index}",
                f"context_agent: shared context path_size={len(traced_edges)} candidate={idx}",
            ],
            "orchestrator": {
                "spawn_count": spawn_count,
                "finished_subtasks": spawn_count,
                "critical_steps": max(1, len(traced_edges)),
                "breadth": min(cfg.swarm_v2.generator_swarm.max_breadth, spawn_count),
                "depth": min(cfg.swarm_v2.generator_swarm.max_depth, 1 if len(traced_edges) <= 2 else 2),
            },
        }
        completion_texts.append(json.dumps(payload, sort_keys=True))
    return completion_texts


def _sample_swarm_v2_completion_texts_with_model(
    env: OSINTEnvironment,
    cfg: SelfPlayTrainingConfig,
    model_name_or_path: str,
    prompts: list[str],
    count: int,
    seen_questions: list[str],
) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if count <= 0:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()

    device = next(model.parameters()).device
    completions: list[str] = []
    validator = SwarmV2ReplayValidator(
        graph=env.graph,
        validation=cfg.swarm_v2.validation,
        shared_context=cfg.swarm_v2.shared_context,
        seen_questions=seen_questions,
    )
    for prompt in prompts:
        if len(completions) >= count:
            break
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}

        best_completion = ""
        best_score = -999
        for attempt_idx, (temperature, top_p) in enumerate([(0.7, 0.9), (0.5, 0.85), (0.3, 0.8)]):
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max(64, int(cfg.generated_task_max_new_tokens)),
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completion_ids = output[0][encoded["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            candidate = parse_generated_task_completion(
                completion,
                max_support_edges=cfg.swarm_v2.validation.max_support_edges,
            )
            validation = validator.validate(candidate)
            score = int(bool(candidate.question)) + int(bool(candidate.answer)) + len(candidate.supporting_edges)
            if validation.is_valid:
                print(f"[self_play][generation_retry] valid_completion attempt={attempt_idx + 1}")
                best_completion = completion
                break
            if score > best_score:
                best_score = score
                best_completion = completion
        completions.append(best_completion)
    return completions


def _materialize_swarm_v2_completions(
    env: OSINTEnvironment,
    cfg: SelfPlayTrainingConfig,
    completion_texts: list[str],
    round_index: int,
    seen_questions: list[str],
    prompt_canonical_candidates: list[dict[str, Any]] | None = None,
) -> tuple[list[TaskInstance], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validator = SwarmV2ReplayValidator(
        graph=env.graph,
        validation=cfg.swarm_v2.validation,
        shared_context=cfg.swarm_v2.shared_context,
        seen_questions=seen_questions,
    )

    tasks: list[TaskInstance] = []
    validation_reports: list[dict[str, Any]] = []
    canonical_graph_candidates: list[dict[str, Any]] = []
    replay_traces: list[dict[str, Any]] = []

    for completion_idx, completion_text in enumerate(completion_texts):
        use_fixed_canonical = str(cfg.canonical_graph_mode).strip().lower() == "fixed"
        if use_fixed_canonical and prompt_canonical_candidates and completion_idx >= len(prompt_canonical_candidates):
            break

        candidate = parse_generated_task_completion(
            completion_text,
            max_support_edges=cfg.swarm_v2.validation.max_support_edges,
        )
        validation = validator.validate(candidate)
        replay_edges = list(validation.replayed_edges or candidate.supporting_edges)
        materialized_tool_trace = _serialize_tool_trace(candidate.tool_trace)
        if not materialized_tool_trace and replay_edges:
            materialized_tool_trace = build_swarm_v2_tool_trace(env.graph, replay_edges)

        if use_fixed_canonical and prompt_canonical_candidates and completion_idx < len(prompt_canonical_candidates):
            canonical_graph = dict(prompt_canonical_candidates[completion_idx])
        else:
            if candidate.canonical_edges or candidate.canonical_nodes:
                canonical_edges = list(candidate.canonical_edges or candidate.supporting_edges)
                canonical_nodes = list(candidate.canonical_nodes)
                if not canonical_nodes:
                    canonical_nodes = sorted(
                        {edge.src for edge in canonical_edges} | {edge.dst for edge in canonical_edges}
                    )
                canonical_graph = {
                    "nodes": canonical_nodes,
                    "edges": [
                        {
                            "src": edge.src,
                            "rel": edge.rel,
                            "dst": edge.dst,
                            "confidence": float(edge.confidence),
                        }
                        for edge in canonical_edges
                    ],
                }
            else:
                canonical_graph = build_swarm_v2_canonical_subgraph(
                    env.graph,
                    candidate.supporting_edges,
                    max_extra_edges=max(0, cfg.swarm_v2.shared_context.max_edges - len(candidate.supporting_edges)),
                )

        canonical_graph_candidates.append(
            {
                "candidate_index": completion_idx,
                "canonical_graph": canonical_graph,
                "question": candidate.question,
                "answer": candidate.answer,
            }
        )
        replay_traces.append(
            {
                "candidate_index": completion_idx,
                "question": candidate.question,
                "tool_trace": materialized_tool_trace,
                "replayed_edges": validation.to_dict()["replayed_edges"],
            }
        )
        validation_reports.append(
            {
                "candidate_index": completion_idx,
                "question": candidate.question,
                "answer": candidate.answer,
                "task_type": candidate.task_type,
                "validation": validation.to_dict(),
                "raw_completion": completion_text,
            }
        )

        if not validation.is_valid:
            continue
        if len(tasks) >= max(1, cfg.generated_tasks_per_round):
            continue

        metadata = {
            "generated_by": "swarm_v2_generator",
            "round": round_index,
            "difficulty": "hard",
            "scenario": "swarm_v2_trace",
            "canonical_graph": canonical_graph,
            "tool_trace": materialized_tool_trace,
            "subagent_outputs": list(candidate.subagent_outputs),
            "validation": validation.to_dict(),
            "shared_context_budget": {
                "max_nodes": cfg.swarm_v2.shared_context.max_nodes,
                "max_edges": cfg.swarm_v2.shared_context.max_edges,
                "target_pressure": cfg.swarm_v2.shared_context.target_pressure,
            },
            "grader": {
                "type": "difficulty_exact_match",
                "answer_type": "node_id",
                "case_sensitive": True,
                "reward_profile": "hard",
            },
        }
        tasks.append(
            TaskInstance(
                task_id=f"swarm_v2_r{round_index}_{len(tasks)}",
                task_type=candidate.task_type or "swarm_v2_trace",
                question=candidate.question,
                answer=candidate.answer,
                supporting_edges=replay_edges,
                metadata=metadata,
            )
        )
        validator.remember(candidate.question)

    return tasks, validation_reports, canonical_graph_candidates, replay_traces


def _run_adversarial_self_play_swarm_v2(
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
    checkpoint_repo_id = _default_hf_checkpoint_repo_id(run_dir)
    if checkpoint_repo_id and _resolve_hf_upload_token():
        print(f"[self_play][hf_upload] checkpoint uploads enabled -> {checkpoint_repo_id}")
    else:
        print("[self_play][hf_upload] checkpoint uploads disabled; set HF token and/or OSINT_HF_CHECKPOINT_REPO_ID.")

    env = OSINTEnvironment(env_config, llm=build_llm_client(env_config.llm))
    seed_tasks = list(env.tasks)
    seed_questions = [task.question for task in seed_tasks]
    generator_model, answerer_model = _resolve_initial_models(training_config)
    initial_generator_model = str(generator_model)
    initial_answerer_model = str(answerer_model)
    rng = random.Random(env_config.seed)

    bootstrap_completions = _fallback_swarm_v2_completion_texts(
        env=env,
        cfg=training_config,
        round_index=0,
        rng=rng,
    )
    rolling_generated_tasks, _, _, _ = _materialize_swarm_v2_completions(
        env=env,
        cfg=training_config,
        completion_texts=bootstrap_completions,
        round_index=0,
        seen_questions=seed_questions,
    )
    if not rolling_generated_tasks:
        rolling_generated_tasks = list(seed_tasks[: max(1, training_config.generated_tasks_per_round)])

    rounds_payload: list[dict[str, Any]] = []

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
            answerer_pre_rows = _build_swarm_v2_answerer_rows(env, answerer_pre_tasks, training_config)
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
                answerer_pre_report_to, answerer_pre_run_name = _resolve_reporting(
                    training_config=training_config,
                    phase_name="answerer-pre",
                    round_index=round_index,
                )
                answerer_pre_reward = AnswererRewardFunction(
                    graph=env.graph,
                    pipeline_mode="swarm_v2",
                    parl_max_parallel_hint=training_config.swarm_v2.answerer_swarm.max_agents,
                )
                answerer_pre_train_result = _train_grpo_phase(
                    model_name_or_path=answerer_model,
                    phase=training_config.answerer_phase,
                    rows=answerer_pre_rows,
                    reward_function=answerer_pre_reward,
                    output_dir=round_dir / f"{training_config.answerer_phase.output_subdir}_pre",
                    tuning_mode=tuning_mode,
                    lora=training_config.lora,
                    report_to=answerer_pre_report_to,
                    run_name=answerer_pre_run_name,
                )
                _maybe_upload_folder_to_hf(
                    round_dir / f"{training_config.answerer_phase.output_subdir}_pre",
                    run_dir,
                    f"Upload answerer-pre checkpoints for round {round_index:03d}",
                )
                answerer_model = str(answerer_pre_train_result["model_path"])
                if topology == "shared":
                    generator_model = answerer_model

        generator_rows, prompt_canonical_candidates = _build_swarm_v2_generator_rows(env, training_config, rng)
        generator_dataset_path = round_dir / "generator_dataset.json"
        _save_rows(generator_dataset_path, generator_rows)

        generator_train_result: dict[str, Any] = {
            "model_path": generator_model,
            "global_step": 0,
            "training_loss": 0.0,
            "train_rows": len(generator_rows),
            "skipped": effective_dry_run,
            "tuning_mode": tuning_mode,
            "frozen_answerer_model": answerer_model,
        }

        if not effective_dry_run:
            generator_report_to, generator_run_name = _resolve_reporting(
                training_config=training_config,
                phase_name="generator",
                round_index=round_index,
            )
            generator_reward = GeneratorRewardFunction(
                graph=env.graph,
                answerer_judge=AnswererJudge(
                    model_name_or_path=answerer_model,
                    max_new_tokens=training_config.answerer_judge_max_new_tokens,
                ),
                weights=training_config.generator_reward_weights,
                max_support_edges=training_config.swarm_v2.validation.max_support_edges,
                pipeline_mode="swarm_v2",
                swarm_v2_validation=training_config.swarm_v2.validation,
                swarm_v2_shared_context=training_config.swarm_v2.shared_context,
                parl_max_parallel_hint=training_config.swarm_v2.generator_swarm.max_agents,
            )
            generator_train_result = _train_grpo_phase(
                model_name_or_path=generator_model,
                phase=training_config.generator_phase,
                rows=generator_rows,
                reward_function=generator_reward,
                output_dir=round_dir / training_config.generator_phase.output_subdir,
                tuning_mode=tuning_mode,
                lora=training_config.lora,
                report_to=generator_report_to,
                run_name=generator_run_name,
            )
            _maybe_upload_folder_to_hf(
                round_dir / training_config.generator_phase.output_subdir,
                run_dir,
                f"Upload generator checkpoints for round {round_index:03d}",
            )
            generator_model = str(generator_train_result["model_path"])
            if topology == "shared":
                answerer_model = generator_model

        if effective_dry_run:
            completion_texts = _fallback_swarm_v2_completion_texts(
                env=env,
                cfg=training_config,
                round_index=round_index,
                rng=rng,
            )
        else:
            completion_texts = _sample_swarm_v2_completion_texts_with_model(
                env=env,
                cfg=training_config,
                model_name_or_path=generator_model,
                prompts=[row["prompt"] for row in generator_rows],
                count=max(1, training_config.generated_tasks_per_round * 2),
                seen_questions=seed_questions + [task.question for task in rolling_generated_tasks],
            )
            if not completion_texts:
                completion_texts = _fallback_swarm_v2_completion_texts(
                    env=env,
                    cfg=training_config,
                    round_index=round_index,
                    rng=rng,
                )

        generated_tasks, validation_reports, canonical_graph_candidates, replay_traces = _materialize_swarm_v2_completions(
            env=env,
            cfg=training_config,
            completion_texts=completion_texts,
            round_index=round_index,
            seen_questions=seed_questions + [task.question for task in rolling_generated_tasks],
            prompt_canonical_candidates=prompt_canonical_candidates,
        )
        if not generated_tasks:
            fallback_completions = _fallback_swarm_v2_completion_texts(
                env=env,
                cfg=training_config,
                round_index=round_index,
                rng=rng,
            )
            generated_tasks, validation_reports, canonical_graph_candidates, replay_traces = _materialize_swarm_v2_completions(
                env=env,
                cfg=training_config,
                completion_texts=fallback_completions,
                round_index=round_index,
                seen_questions=seed_questions + [task.question for task in rolling_generated_tasks],
                prompt_canonical_candidates=None,
            )

        if generated_tasks:
            rolling_generated_tasks = list(generated_tasks)

        canonical_graph_candidates_path = round_dir / "canonical_graph_candidates.json"
        replay_traces_path = round_dir / "replay_traces.json"
        validation_reports_path = round_dir / "validation_reports.json"
        generated_tasks_path = round_dir / "generated_tasks.json"
        _save_payload(canonical_graph_candidates_path, prompt_canonical_candidates or canonical_graph_candidates)
        _save_payload(replay_traces_path, replay_traces)
        _save_payload(validation_reports_path, validation_reports)
        _save_tasks(generated_tasks_path, generated_tasks)

        answerer_tasks = _select_answerer_tasks(
            seed_tasks=seed_tasks,
            generated_tasks=generated_tasks,
            cfg=training_config,
            rng=rng,
        )
        answerer_rows = _build_swarm_v2_answerer_rows(env, answerer_tasks, training_config)
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
            answerer_report_to, answerer_run_name = _resolve_reporting(
                training_config=training_config,
                phase_name="answerer",
                round_index=round_index,
            )
            answerer_reward = AnswererRewardFunction(
                graph=env.graph,
                pipeline_mode="swarm_v2",
                parl_max_parallel_hint=training_config.swarm_v2.answerer_swarm.max_agents,
            )
            answerer_train_result = _train_grpo_phase(
                model_name_or_path=answerer_model,
                phase=training_config.answerer_phase,
                rows=answerer_rows,
                reward_function=answerer_reward,
                output_dir=round_dir / training_config.answerer_phase.output_subdir,
                tuning_mode=tuning_mode,
                lora=training_config.lora,
                report_to=answerer_report_to,
                run_name=answerer_run_name,
            )
            _maybe_upload_folder_to_hf(
                round_dir / training_config.answerer_phase.output_subdir,
                run_dir,
                f"Upload answerer checkpoints for round {round_index:03d}",
            )
            answerer_model = str(answerer_train_result["model_path"])
            if topology == "shared":
                generator_model = answerer_model

        rounds_payload.append(
            {
                "round": round_index,
                "dry_run": effective_dry_run,
                "pipeline_mode": "swarm_v2",
                "phase_schedule": phase_schedule,
                "generator": generator_train_result,
                "answerer": answerer_train_result,
                "answerer_pre": answerer_pre_train_result,
                "generated_task_count": len(generated_tasks),
                "answerer_task_count": len(answerer_tasks),
                "answerer_pre_task_count": len(answerer_pre_tasks),
                "artifacts": {
                    "generator_dataset": str(generator_dataset_path),
                    "answerer_dataset": str(answerer_dataset_path),
                    "generated_tasks": str(generated_tasks_path),
                    "canonical_graph_candidates": str(canonical_graph_candidates_path),
                    "replay_traces": str(replay_traces_path),
                    "validation_reports": str(validation_reports_path),
                    "answerer_pre_dataset": str(answerer_pre_dataset_path) if answerer_pre_dataset_path else "",
                },
            }
        )

    post_training_evaluation = _run_post_training_evaluation(
        env_config=env_config,
        training_config=training_config,
        generator_model=generator_model,
        answerer_models={
            "finetuned_answerer": answerer_model,
            "original_answerer": initial_answerer_model,
        },
        output_dir=run_dir,
        pipeline_mode="swarm_v2",
        effective_dry_run=effective_dry_run,
    )
    final_payload = {
        "dry_run": effective_dry_run,
        "pipeline_mode": "swarm_v2",
        "output_dir": str(run_dir),
        "model_topology": topology,
        "phase_schedule": phase_schedule,
        "tuning_mode": tuning_mode,
        "canonical_graph_mode": str(training_config.canonical_graph_mode).strip().lower() or "generate",
        "rounds": rounds_payload,
        "final_models": {
            "generator": generator_model,
            "answerer": answerer_model,
        },
        "initial_models": {
            "generator": initial_generator_model,
            "answerer": initial_answerer_model,
        },
        "post_training_evaluation": post_training_evaluation,
        "kimi_objective_mapping": {
            "grouped_rollouts": "TRL GRPO num_generations",
            "mean_centered_advantage": "GRPO relative reward baseline",
            "token_level_clipping": "GRPO epsilon clipping over policy ratios",
            "reference_regularization": "GRPO beta KL term",
            "toggle_self_play": "Alternating generator and answerer rounds",
            "parallel_orchestration": "PARL-inspired auxiliary reward over generator and answerer swarms",
        },
    }

    summary_path = run_dir / "self_play_summary.json"
    summary_path.write_text(json.dumps(final_payload, indent=2, sort_keys=True), encoding="utf-8")
    final_payload["summary_path"] = str(summary_path)
    _maybe_upload_file_to_hf(summary_path, run_dir, "Upload self-play summary")
    _maybe_upload_file_to_hf(run_dir / "post_training_evaluation.json", run_dir, "Upload post-training evaluation")
    return final_payload



def run_adversarial_self_play(
    env_config: EnvironmentConfig,
    training_config: SelfPlayTrainingConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    if str(training_config.pipeline_mode).strip().lower() == "swarm_v2":
        return _run_adversarial_self_play_swarm_v2(
            env_config=env_config,
            training_config=training_config,
            dry_run=dry_run,
        )

    effective_dry_run = bool(dry_run or training_config.dry_run)
    topology = str(training_config.model_topology).strip().lower() or "dual"
    phase_schedule = str(training_config.phase_schedule).strip().lower() or "generator_answerer"
    tuning_mode = str(training_config.tuning_mode).strip().lower() or "full"

    run_dir = Path(training_config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_repo_id = _default_hf_checkpoint_repo_id(run_dir)
    if checkpoint_repo_id and _resolve_hf_upload_token():
        print(f"[self_play][hf_upload] checkpoint uploads enabled -> {checkpoint_repo_id}")
    else:
        print("[self_play][hf_upload] checkpoint uploads disabled; set HF token and/or OSINT_HF_CHECKPOINT_REPO_ID.")

    env = OSINTEnvironment(env_config, llm=build_llm_client(env_config.llm))
    seed_tasks = list(env.tasks)

    generator_model, answerer_model = _resolve_initial_models(training_config)
    initial_generator_model = str(generator_model)
    initial_answerer_model = str(answerer_model)

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
                answerer_pre_report_to, answerer_pre_run_name = _resolve_reporting(
                    training_config=training_config,
                    phase_name="answerer-pre",
                    round_index=round_index,
                )
                answerer_pre_reward = AnswererRewardFunction(graph=env.graph)
                answerer_pre_train_result = _train_grpo_phase(
                    model_name_or_path=answerer_model,
                    phase=training_config.answerer_phase,
                    rows=answerer_pre_rows,
                    reward_function=answerer_pre_reward,
                    output_dir=round_dir / f"{training_config.answerer_phase.output_subdir}_pre",
                    tuning_mode=tuning_mode,
                    lora=training_config.lora,
                    report_to=answerer_pre_report_to,
                    run_name=answerer_pre_run_name,
                )
                _maybe_upload_folder_to_hf(
                    round_dir / f"{training_config.answerer_phase.output_subdir}_pre",
                    run_dir,
                    f"Upload answerer-pre checkpoints for round {round_index:03d}",
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
            generator_report_to, generator_run_name = _resolve_reporting(
                training_config=training_config,
                phase_name="generator",
                round_index=round_index,
            )
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
                report_to=generator_report_to,
                run_name=generator_run_name,
            )
            _maybe_upload_folder_to_hf(
                round_dir / training_config.generator_phase.output_subdir,
                run_dir,
                f"Upload generator checkpoints for round {round_index:03d}",
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
                max_new_tokens=training_config.generated_task_max_new_tokens,
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
            answerer_report_to, answerer_run_name = _resolve_reporting(
                training_config=training_config,
                phase_name="answerer",
                round_index=round_index,
            )
            answerer_reward = AnswererRewardFunction(graph=env.graph)
            answerer_train_result = _train_grpo_phase(
                model_name_or_path=answerer_model,
                phase=training_config.answerer_phase,
                rows=answerer_rows,
                reward_function=answerer_reward,
                output_dir=round_dir / training_config.answerer_phase.output_subdir,
                tuning_mode=tuning_mode,
                lora=training_config.lora,
                report_to=answerer_report_to,
                run_name=answerer_run_name,
            )
            _maybe_upload_folder_to_hf(
                round_dir / training_config.answerer_phase.output_subdir,
                run_dir,
                f"Upload answerer checkpoints for round {round_index:03d}",
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
                "pipeline_mode": "legacy",
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

    post_training_evaluation = _run_post_training_evaluation(
        env_config=env_config,
        training_config=training_config,
        generator_model=generator_model,
        answerer_models={
            "finetuned_answerer": answerer_model,
            "original_answerer": initial_answerer_model,
        },
        output_dir=run_dir,
        pipeline_mode="legacy",
        effective_dry_run=effective_dry_run,
    )
    final_payload = {
        "dry_run": effective_dry_run,
        "pipeline_mode": "legacy",
        "output_dir": str(run_dir),
        "model_topology": topology,
        "phase_schedule": phase_schedule,
        "tuning_mode": tuning_mode,
        "canonical_graph_mode": str(training_config.canonical_graph_mode).strip().lower() or "generate",
        "rounds": rounds_payload,
        "final_models": {
            "generator": generator_model,
            "answerer": answerer_model,
        },
        "initial_models": {
            "generator": initial_generator_model,
            "answerer": initial_answerer_model,
        },
        "post_training_evaluation": post_training_evaluation,
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
    _maybe_upload_file_to_hf(summary_path, run_dir, "Upload self-play summary")
    _maybe_upload_file_to_hf(run_dir / "post_training_evaluation.json", run_dir, "Upload post-training evaluation")

    return final_payload
