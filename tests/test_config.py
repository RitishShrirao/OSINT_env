import json
from pathlib import Path

from osint_env.config.shared import load_seeding_config, load_shared_config


def test_shared_config_defaults_when_file_missing():
    config = load_shared_config("/tmp/does_not_exist_for_osint_config.json")
    assert config.environment.max_steps > 0
    assert config.runtime.default_episodes > 0


def test_shared_config_parses_swarm_and_seeding(tmp_path: Path):
    path = tmp_path / "shared.json"
    path.write_text(
        json.dumps(
            {
                "environment": {"seed": 19, "max_steps": 9},
                "swarm": {"enabled": True, "max_agents": 3, "max_breadth": 2, "max_width": 2, "max_depth": 2},
                "seeding": {
                    "seeded_questions": [
                        {
                            "question": "Which canonical user owns alias alias_seed_001?",
                            "answer": "user_seed_001",
                        }
                    ],
                    "llm_generation_parallel": True,
                    "llm_generation_workers": 4,
                    "llm_generation_retries": 3,
                    "allow_template_fallback_on_llm_failure": False
                },
                "runtime": {"default_episodes": 5},
                "llm": {"provider": "ollama", "model": "qwen3:2b", "timeout_seconds": 333},
            }
        ),
        encoding="utf-8",
    )

    config = load_shared_config(path)
    assert config.environment.seed == 19
    assert config.environment.swarm.enabled is True
    assert config.environment.swarm.max_width == 2
    assert len(config.environment.seeding.seeded_questions) == 1
    assert config.runtime.default_episodes == 5
    assert config.environment.llm.provider == "ollama"
    assert config.environment.llm.model == "qwen3:2b"
    assert config.environment.llm.timeout_seconds == 333
    assert config.environment.seeding.llm_generation_parallel is True
    assert config.environment.seeding.llm_generation_workers == 4
    assert config.environment.seeding.llm_generation_retries == 3
    assert config.environment.seeding.allow_template_fallback_on_llm_failure is False


def test_load_seeding_config_supports_top_level_object(tmp_path: Path):
    path = tmp_path / "seeding.json"
    path.write_text(
        json.dumps(
            {
                "seeded_nodes": [
                    {"node_id": "alias_seed_1", "node_type": "alias", "attrs": {"handle": "@seed"}},
                    {"node_id": "user_seed_1", "node_type": "user", "attrs": {"name": "Seed"}},
                ],
                "seeded_edges": [{"src": "alias_seed_1", "rel": "alias_of", "dst": "user_seed_1"}],
                "seeded_questions": [{"question": "Which canonical user owns alias alias_seed_1?", "answer": "user_seed_1"}],
            }
        ),
        encoding="utf-8",
    )

    seeding = load_seeding_config(path)
    assert len(seeding.seeded_nodes) == 2
    assert len(seeding.seeded_edges) == 1
    assert seeding.seeded_questions[0].answer == "user_seed_1"
