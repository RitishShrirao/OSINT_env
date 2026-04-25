from pathlib import Path
import json

from osint_env.training.config import load_self_play_config


def test_self_play_config_defaults_when_missing():
    cfg = load_self_play_config("/tmp/does_not_exist_for_self_play_config.json")
    assert cfg.rounds >= 1
    assert cfg.pipeline_mode in {"legacy", "swarm_v2"}
    assert cfg.model_topology in {"dual", "shared"}
    assert cfg.phase_schedule in {"generator_answerer", "answerer_generator_answerer"}
    assert cfg.tuning_mode in {"full", "lora"}
    assert cfg.generator_phase.max_steps >= 1
    assert cfg.answerer_phase.max_steps >= 1
    assert cfg.generator_reward_weights.hardness > 0.0
    assert cfg.swarm_v2.generator_swarm.shared_context is True
    assert cfg.swarm_v2.validation.max_support_edges >= 1


def test_self_play_config_parses_overrides(tmp_path: Path):
    cfg_path = tmp_path / "self_play.json"
    cfg_path.write_text(
        json.dumps(
            {
                "rounds": 5,
                "output_dir": "artifacts/custom_self_play",
                "dry_run": False,
                "pipeline_mode": "swarm_v2",
                "model_topology": "shared",
                "phase_schedule": "answerer_generator_answerer",
                "tuning_mode": "lora",
                "shared_model_name_or_path": "/models/local-base",
                "seed_tasks_per_round": 12,
                "generated_tasks_per_round": 18,
                "swarm_v2": {
                    "generator_swarm": {
                        "shared_context": True,
                        "max_agents": 5,
                        "max_breadth": 4,
                        "max_depth": 3,
                        "planner_rounds": 3,
                        "tools_per_agent": 2,
                    },
                    "answerer_swarm": {
                        "shared_context": True,
                        "max_agents": 4,
                        "max_breadth": 3,
                        "max_depth": 2,
                        "planner_rounds": 2,
                        "tools_per_agent": 2,
                    },
                    "validation": {
                        "max_support_edges": 6,
                        "max_path_hops": 3,
                        "max_context_nodes": 10,
                        "max_context_edges": 6,
                        "duplicate_similarity_threshold": 0.75,
                    },
                    "shared_context": {
                        "shared_by_default": True,
                        "max_nodes": 10,
                        "max_edges": 6,
                        "target_pressure": 0.9,
                    },
                },
                "generator_reward_weights": {
                    "validity": 0.2,
                    "hardness": 0.6,
                    "diversity": 0.1,
                    "consistency": 0.1,
                },
                "lora": {
                    "r": 32,
                    "alpha": 64,
                    "dropout": 0.1,
                    "target_modules": ["q_proj", "v_proj"],
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "generator_phase": {
                    "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
                    "max_steps": 77,
                    "num_generations": 6,
                    "loss_type": "grpo",
                    "scale_rewards": "group",
                    "output_subdir": "gen_phase",
                },
                "answerer_phase": {
                    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "max_steps": 55,
                    "num_generations": 5,
                    "output_subdir": "ans_phase",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_self_play_config(cfg_path)
    assert cfg.rounds == 5
    assert cfg.output_dir == "artifacts/custom_self_play"
    assert cfg.dry_run is False
    assert cfg.pipeline_mode == "swarm_v2"
    assert cfg.model_topology == "shared"
    assert cfg.phase_schedule == "answerer_generator_answerer"
    assert cfg.tuning_mode == "lora"
    assert cfg.shared_model_name_or_path == "/models/local-base"
    assert cfg.seed_tasks_per_round == 12
    assert cfg.generated_tasks_per_round == 18
    assert cfg.swarm_v2.generator_swarm.max_agents == 5
    assert cfg.swarm_v2.answerer_swarm.max_agents == 4
    assert cfg.swarm_v2.validation.max_support_edges == 6
    assert cfg.swarm_v2.shared_context.target_pressure == 0.9
    assert cfg.generator_reward_weights.hardness == 0.6
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 64
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]

    assert cfg.generator_phase.model_name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.generator_phase.max_steps == 77
    assert cfg.generator_phase.num_generations == 6
    assert cfg.generator_phase.loss_type == "grpo"
    assert cfg.generator_phase.scale_rewards == "group"
    assert cfg.generator_phase.output_subdir == "gen_phase"

    assert cfg.answerer_phase.model_name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.answerer_phase.max_steps == 55
    assert cfg.answerer_phase.num_generations == 5
    assert cfg.answerer_phase.output_subdir == "ans_phase"


def test_self_play_config_keeps_legacy_mode_when_not_set(tmp_path: Path):
    cfg_path = tmp_path / "legacy_self_play.json"
    cfg_path.write_text(
        json.dumps(
            {
                "rounds": 2,
                "max_support_edges": 11,
            }
        ),
        encoding="utf-8",
    )

    cfg = load_self_play_config(cfg_path)
    assert cfg.pipeline_mode == "legacy"
    assert cfg.max_support_edges == 11
    assert cfg.swarm_v2.validation.max_support_edges == 11
