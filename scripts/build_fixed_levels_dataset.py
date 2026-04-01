from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.data.generator import DatasetGenerator
from osint_env.domain.models import Edge, TaskInstance
from osint_env.llm import build_llm_client


def edge_to_dict(edge: Edge) -> dict[str, Any]:
    return {
        "src": edge.src,
        "rel": edge.rel,
        "dst": edge.dst,
        "confidence": float(edge.confidence),
    }


def task_to_dict(task: TaskInstance) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "question": task.question,
        "answer": task.answer,
        "supporting_edges": [edge_to_dict(e) for e in task.supporting_edges],
        "metadata": dict(task.metadata),
    }


def build_fixed_snapshot(seed_path: Path) -> dict[str, Any]:
    seeding = load_seeding_config(seed_path)
    fixed_nodes = []
    for node in seeding.seeded_nodes:
        fixed_nodes.append(
            {
                "node_id": node.node_id,
                "node_type": str(getattr(node.node_type, "value", node.node_type)),
                "attrs": dict(node.attrs),
            }
        )
    fixed_edges = [
        {
            "src": edge.src,
            "rel": edge.rel,
            "dst": edge.dst,
            "confidence": float(edge.confidence),
        }
        for edge in seeding.seeded_edges
    ]
    fixed_questions = []
    for idx, q in enumerate(seeding.seeded_questions):
        fixed_questions.append(
            {
                "task_id": f"fixed_task_{idx:02d}",
                "task_type": q.task_type,
                "question": q.question,
                "answer": q.answer,
                "supporting_edges": [
                    {
                        "src": edge.src,
                        "rel": edge.rel,
                        "dst": edge.dst,
                        "confidence": float(edge.confidence),
                    }
                    for edge in q.supporting_edges
                ],
                "metadata": dict(q.metadata),
            }
        )

    difficulty_counts = Counter(str(q.get("metadata", {}).get("difficulty", "unknown")) for q in fixed_questions)
    return {
        "dataset_name": "fixed_levels_submission_set",
        "source_seed": str(seed_path),
        "graph": {
            "nodes": fixed_nodes,
            "edges": fixed_edges,
            "node_count": len(fixed_nodes),
            "edge_count": len(fixed_edges),
        },
        "questions": fixed_questions,
        "question_count": len(fixed_questions),
        "difficulty_counts": dict(difficulty_counts),
    }


def build_complete_snapshot(shared_config_path: Path, seed_path: Path) -> dict[str, Any]:
    shared = load_shared_config(shared_config_path)
    env_cfg = clone_environment_config(shared.environment)
    env_cfg.seeding = load_seeding_config(seed_path)

    llm_client = build_llm_client(env_cfg.llm)
    generator = DatasetGenerator(config=env_cfg, llm=llm_client)

    graph = generator.build_canonical_graph()
    views = generator.build_platform_views(graph)
    tasks = generator.generate_tasks(graph, views, count=max(15, len(env_cfg.seeding.seeded_questions)))

    difficulty_counts = Counter(str(task.metadata.get("difficulty", "unknown")) for task in tasks)

    return {
        "dataset_name": "fixed_levels_submission_set",
        "generation_mode": "llm_expanded",
        "shared_config": str(shared_config_path),
        "seed_file": str(seed_path),
        "llm": asdict(env_cfg.llm),
        "environment": {
            "n_users": env_cfg.n_users,
            "alias_density": env_cfg.alias_density,
            "noise_level": env_cfg.noise_level,
            "red_herring_rate": env_cfg.red_herring_rate,
            "seed": env_cfg.seed,
        },
        "canonical_graph": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "attrs": dict(node.attrs),
                }
                for node in sorted(graph.nodes.values(), key=lambda n: n.node_id)
            ],
            "edges": [edge_to_dict(edge) for edge in graph.edges],
        },
        "platform_views": {
            "microblog_posts": views.microblog_posts,
            "forum_threads": views.forum_threads,
            "profiles": views.profiles,
            "counts": {
                "microblog_posts": len(views.microblog_posts),
                "forum_threads": len(views.forum_threads),
                "profiles": len(views.profiles),
            },
        },
        "tasks": [task_to_dict(task) for task in tasks],
        "task_count": len(tasks),
        "difficulty_counts": dict(difficulty_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed difficulty dataset artifacts.")
    parser.add_argument(
        "--seed-file",
        default="datasets/fixed_levels/seed_fixed_levels.json",
        help="Path to seeding JSON with fixed graph/questions.",
    )
    parser.add_argument(
        "--shared-config",
        default="datasets/fixed_levels/shared_config_fixed_levels.json",
        help="Path to shared config used for LLM-expanded generation.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/fixed_levels",
        help="Directory where dataset artifacts are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_path = Path(args.seed_file)
    shared_path = Path(args.shared_config)

    fixed_snapshot = build_fixed_snapshot(seed_path)
    fixed_path = output_dir / "fixed_graph_questions.json"
    fixed_path.write_text(json.dumps(fixed_snapshot, indent=2, sort_keys=True), encoding="utf-8")

    complete_snapshot = build_complete_snapshot(shared_path, seed_path)
    complete_path = output_dir / "complete_dataset_qwen_generated.json"
    complete_path.write_text(json.dumps(complete_snapshot, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "fixed_dataset": str(fixed_path),
        "complete_dataset": str(complete_path),
        "fixed_nodes": fixed_snapshot["graph"]["node_count"],
        "fixed_edges": fixed_snapshot["graph"]["edge_count"],
        "fixed_questions": fixed_snapshot["question_count"],
        "complete_nodes": complete_snapshot["canonical_graph"]["node_count"],
        "complete_edges": complete_snapshot["canonical_graph"]["edge_count"],
        "complete_tasks": complete_snapshot["task_count"],
        "difficulty_counts": complete_snapshot["difficulty_counts"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
