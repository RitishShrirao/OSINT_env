import json
import random
from copy import deepcopy
from pathlib import Path

from osint_env.data.generator import (
    build_swarm_v2_canonical_subgraph,
    build_swarm_v2_path_candidates,
    build_swarm_v2_tool_trace,
    emit_swarm_v2_question,
    select_swarm_v2_answer,
)
from osint_env.domain.models import CanonicalGraph, Edge, EnvironmentConfig, Node, NodeType
from osint_env.env.environment import OSINTEnvironment
from osint_env.training import SelfPlayTrainingConfig, run_adversarial_self_play
from osint_env.training.config import GeneratorRewardWeights
from osint_env.training.rewards import (
    decode_completion_text,
    extract_answer_from_completion,
    GeneratorRewardFunction,
    SwarmV2ReplayValidator,
    parse_generated_task_completion,
)


class DummyJudge:
    def __init__(self, answer: str):
        self._answer = answer

    def answer(self, question: str) -> str:
        del question
        return self._answer


def _edge_payload(edge: Edge) -> dict[str, object]:
    return {
        "src": edge.src,
        "rel": edge.rel,
        "dst": edge.dst,
        "confidence": float(edge.confidence),
    }


def _build_valid_candidate_payload(env: OSINTEnvironment, cfg: SelfPlayTrainingConfig) -> dict[str, object]:
    path_candidates = build_swarm_v2_path_candidates(
        env.graph,
        rng=random.Random(17),
        count=1,
        min_hops=2,
        max_hops=cfg.swarm_v2.validation.max_path_hops,
    )
    assert path_candidates
    path_edges = path_candidates[0]
    question = emit_swarm_v2_question(path_edges)
    answer = select_swarm_v2_answer(path_edges)
    return {
        "canonical_graph": build_swarm_v2_canonical_subgraph(
            env.graph,
            path_edges,
            max_extra_edges=max(0, cfg.swarm_v2.shared_context.max_edges - len(path_edges)),
        ),
        "question": question,
        "answer": answer,
        "task_type": "swarm_v2_trace",
        "supporting_edges": [_edge_payload(edge) for edge in path_edges],
        "tool_trace": build_swarm_v2_tool_trace(env.graph, path_edges),
        "subagent_outputs": [
            f"path_agent_{idx}: {edge.src} --{edge.rel}--> {edge.dst}"
            for idx, edge in enumerate(path_edges)
        ]
        + ["question_agent: deterministic relation-path question"],
        "orchestrator": {
            "spawn_count": 3,
            "finished_subtasks": 3,
            "critical_steps": 2,
            "breadth": 3,
            "depth": 1,
        },
    }


def test_decode_completion_text_handles_nested_content_parts():
    payload = {"question": "Q", "answer": "A", "supporting_edges": []}
    completion = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload),
                }
            ],
        }
    ]

    decoded = decode_completion_text(completion)
    parsed = parse_generated_task_completion(decoded)

    assert decoded == json.dumps(payload)
    assert parsed.question == "Q"
    assert parsed.answer == "A"
    assert parsed.is_valid is True
    assert extract_answer_from_completion(decoded) == "A"


def test_parse_generated_task_completion_prefers_relevant_json_blob():
    payload = {"question": "Q", "answer": "A", "supporting_edges": []}
    completion_text = (
        'Example: {"note": "ignore"}\n'
        f"Final: {json.dumps(payload)}\n"
        '{"trailing": true}'
    )

    parsed = parse_generated_task_completion(completion_text)

    assert parsed.question == "Q"
    assert parsed.answer == "A"
    assert parsed.is_valid is True


def test_swarm_v2_duplicate_check_does_not_reject_distinct_relation_paths():
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    env = OSINTEnvironment(EnvironmentConfig(seed=41, n_users=18, max_steps=6))
    path_candidates = build_swarm_v2_path_candidates(
        env.graph,
        rng=random.Random(19),
        count=8,
        min_hops=2,
        max_hops=cfg.swarm_v2.validation.max_path_hops,
    )
    assert len(path_candidates) >= 2

    payload_a = {
        "canonical_graph": build_swarm_v2_canonical_subgraph(env.graph, path_candidates[0], max_extra_edges=2),
        "question": emit_swarm_v2_question(path_candidates[0]),
        "answer": select_swarm_v2_answer(path_candidates[0]),
        "task_type": "swarm_v2_trace",
        "supporting_edges": [_edge_payload(edge) for edge in path_candidates[0]],
        "tool_trace": build_swarm_v2_tool_trace(env.graph, path_candidates[0]),
        "subagent_outputs": ["path_agent: candidate_a"],
        "orchestrator": {"spawn_count": 2, "finished_subtasks": 2, "critical_steps": 2, "breadth": 2, "depth": 1},
    }
    payload_b = {
        "canonical_graph": build_swarm_v2_canonical_subgraph(env.graph, path_candidates[1], max_extra_edges=2),
        "question": emit_swarm_v2_question(path_candidates[1]),
        "answer": select_swarm_v2_answer(path_candidates[1]),
        "task_type": "swarm_v2_trace",
        "supporting_edges": [_edge_payload(edge) for edge in path_candidates[1]],
        "tool_trace": build_swarm_v2_tool_trace(env.graph, path_candidates[1]),
        "subagent_outputs": ["path_agent: candidate_b"],
        "orchestrator": {"spawn_count": 2, "finished_subtasks": 2, "critical_steps": 2, "breadth": 2, "depth": 1},
    }
    assert payload_a["question"] != payload_b["question"]

    distinct_validator = SwarmV2ReplayValidator(
        graph=env.graph,
        validation=cfg.swarm_v2.validation,
        shared_context=cfg.swarm_v2.shared_context,
        seen_questions=[str(payload_b["question"])],
    )
    distinct_result = distinct_validator.validate(parse_generated_task_completion(json.dumps(payload_a)))
    assert distinct_result.is_valid is True

    duplicate_validator = SwarmV2ReplayValidator(
        graph=env.graph,
        validation=cfg.swarm_v2.validation,
        shared_context=cfg.swarm_v2.shared_context,
        seen_questions=[str(payload_a["question"])],
    )
    duplicate_result = duplicate_validator.validate(parse_generated_task_completion(json.dumps(payload_a)))
    assert duplicate_result.is_valid is False
    assert "duplicate_or_near_duplicate" in duplicate_result.reasons


def test_swarm_v2_replay_validator_accepts_valid_candidate_and_rejects_invalid_cases():
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    env = OSINTEnvironment(EnvironmentConfig(seed=23, n_users=18, max_steps=6))
    payload = _build_valid_candidate_payload(env, cfg)

    validator = SwarmV2ReplayValidator(
        graph=env.graph,
        validation=cfg.swarm_v2.validation,
        shared_context=cfg.swarm_v2.shared_context,
        seen_questions=[],
    )
    valid = validator.validate(parse_generated_task_completion(json.dumps(payload)))
    assert valid.is_valid is True

    leaked_payload = deepcopy(payload)
    leaked_payload["question"] = f"{payload['question']} {payload['answer']}"
    leaked = validator.validate(parse_generated_task_completion(json.dumps(leaked_payload)))
    assert leaked.is_valid is False
    assert "answer_leakage" in leaked.reasons

    no_trace_payload = deepcopy(payload)
    no_trace_payload["tool_trace"] = []
    no_trace = validator.validate(parse_generated_task_completion(json.dumps(no_trace_payload)))
    assert no_trace.is_valid is True
    assert no_trace.replayed_edges

    unseen_payload = deepcopy(payload)
    unseen_payload["supporting_edges"][0]["dst"] = "user_missing"
    unseen = validator.validate(parse_generated_task_completion(json.dumps(unseen_payload)))
    assert unseen.is_valid is False
    assert "unseen_nodes_or_edges" in unseen.reasons


def test_swarm_v2_replay_validator_can_derive_tool_trace_from_support_edges():
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    env = OSINTEnvironment(EnvironmentConfig(seed=27, n_users=18, max_steps=6))
    payload = _build_valid_candidate_payload(env, cfg)
    payload.pop("tool_trace", None)

    validator = SwarmV2ReplayValidator(
        graph=env.graph,
        validation=cfg.swarm_v2.validation,
        shared_context=cfg.swarm_v2.shared_context,
        seen_questions=[],
    )
    result = validator.validate(parse_generated_task_completion(json.dumps(payload)))
    assert result.is_valid is True


def test_swarm_v2_replay_validator_rejects_non_unique_paths():
    graph = CanonicalGraph(
        nodes={
            "user_root": Node("user_root", NodeType.USER, {}),
            "user_mid1": Node("user_mid1", NodeType.USER, {}),
            "user_mid2": Node("user_mid2", NodeType.USER, {}),
            "user_target": Node("user_target", NodeType.USER, {}),
        },
        edges=[
            Edge("user_root", "linked_to", "user_mid1"),
            Edge("user_root", "linked_to", "user_mid2"),
            Edge("user_mid1", "knows", "user_target"),
            Edge("user_mid2", "knows", "user_target"),
        ],
    )
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    ambiguous_path = [
        Edge("user_root", "linked_to", "user_mid1"),
        Edge("user_mid1", "knows", "user_target"),
    ]
    payload = {
        "canonical_graph": build_swarm_v2_canonical_subgraph(graph, ambiguous_path, max_extra_edges=1),
        "question": emit_swarm_v2_question(ambiguous_path),
        "answer": select_swarm_v2_answer(ambiguous_path),
        "task_type": "swarm_v2_trace",
        "supporting_edges": [_edge_payload(edge) for edge in ambiguous_path],
        "tool_trace": build_swarm_v2_tool_trace(graph, ambiguous_path),
        "subagent_outputs": ["path_agent: ambiguous linked_to -> knows trace"],
        "orchestrator": {"spawn_count": 2, "finished_subtasks": 2, "critical_steps": 2, "breadth": 2, "depth": 1},
    }
    validator = SwarmV2ReplayValidator(
        graph=graph,
        validation=cfg.swarm_v2.validation,
        shared_context=cfg.swarm_v2.shared_context,
        seen_questions=[],
    )
    result = validator.validate(parse_generated_task_completion(json.dumps(payload)))
    assert result.is_valid is False
    assert "non_unique_derivation_path" in result.reasons


def test_swarm_v2_generator_reward_prefers_valid_parallel_diverse_tasks():
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    env = OSINTEnvironment(EnvironmentConfig(seed=29, n_users=18, max_steps=6))
    payload = _build_valid_candidate_payload(env, cfg)

    reward_fn = GeneratorRewardFunction(
        graph=env.graph,
        answerer_judge=DummyJudge(answer="wrong_answer"),
        weights=GeneratorRewardWeights(),
        max_support_edges=cfg.swarm_v2.validation.max_support_edges,
        pipeline_mode="swarm_v2",
        swarm_v2_validation=cfg.swarm_v2.validation,
        swarm_v2_shared_context=cfg.swarm_v2.shared_context,
        parl_max_parallel_hint=cfg.swarm_v2.generator_swarm.max_agents,
    )

    spawn_only = deepcopy(payload)
    spawn_only["orchestrator"]["spawn_count"] = 6
    spawn_only["orchestrator"]["finished_subtasks"] = 0
    spawn_only["orchestrator"]["critical_steps"] = 6

    duplicate_workers = deepcopy(payload)
    duplicate_workers["subagent_outputs"] = ["same worker trace"] * 4

    answer_leak = deepcopy(payload)
    answer_leak["question"] = f"{payload['question']} {payload['answer']}"

    overflow = deepcopy(payload)
    overflow["supporting_edges"] = payload["supporting_edges"] + payload["supporting_edges"]

    unsupported_answer = deepcopy(payload)
    unsupported_answer["answer"] = "user_not_in_graph"

    serial_collapse = deepcopy(payload)
    serial_collapse["orchestrator"] = {
        "spawn_count": 1,
        "finished_subtasks": 1,
        "critical_steps": 7,
        "breadth": 1,
        "depth": 1,
    }

    scores = reward_fn(
        completions=[
            json.dumps(payload),
            json.dumps(spawn_only),
            json.dumps(duplicate_workers),
            json.dumps(answer_leak),
            json.dumps(overflow),
            json.dumps(unsupported_answer),
            json.dumps(serial_collapse),
        ]
    )

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > scores[6]
    assert scores[3] < 0
    assert scores[4] < 0
    assert scores[5] < 0


def test_swarm_v2_generator_reward_grades_invalid_outputs_instead_of_constant_penalty():
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    env = OSINTEnvironment(EnvironmentConfig(seed=31, n_users=18, max_steps=6))
    valid_payload = _build_valid_candidate_payload(env, cfg)

    reward_fn = GeneratorRewardFunction(
        graph=env.graph,
        answerer_judge=DummyJudge(answer="wrong_answer"),
        weights=GeneratorRewardWeights(),
        max_support_edges=cfg.swarm_v2.validation.max_support_edges,
        pipeline_mode="swarm_v2",
        swarm_v2_validation=cfg.swarm_v2.validation,
        swarm_v2_shared_context=cfg.swarm_v2.shared_context,
        parl_max_parallel_hint=cfg.swarm_v2.generator_swarm.max_agents,
    )

    missing_everything = "not json"
    partial_json = json.dumps({"question": "Who is linked by this path?", "answer": valid_payload["answer"]})
    partial_edges = json.dumps(
        {
            "question": valid_payload["question"],
            "answer": valid_payload["answer"],
            "supporting_edges": valid_payload["supporting_edges"],
        }
    )

    scores = reward_fn(completions=[missing_everything, partial_json, partial_edges, json.dumps(valid_payload)])

    assert len(set(scores)) > 2
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]
    assert scores[3] != scores[0]
    assert reward_fn._debug_last_batch["batch_reward_std"] > 0.0
    assert reward_fn._debug_last_batch["valid_output_ratio"] == 0.25


def test_parse_generated_task_completion_handles_garbage_orchestrator_values():
    """Regression: model emits e.g. ``"none"`` for an orchestrator integer.

    Previously this crashed the GRPO trainer with
    ``ValueError: invalid literal for int() with base 10: 'none'``.
    """
    completion = json.dumps(
        {
            "question": "Q?",
            "answer": "user_7",
            "supporting_edges": [
                {"src": "a", "rel": "knows", "dst": "user_7", "confidence": 1.0}
            ],
            "tool_trace": [],
            "orchestrator": {
                "spawn_count": "none",
                "finished_subtasks": "N/A",
                "critical_steps": True,
                "breadth": "2 agents",
                "depth": None,
            },
        }
    )

    candidate = parse_generated_task_completion(completion)
    assert candidate.orchestrator.spawn_count == 0
    assert candidate.orchestrator.finished_subtasks == 0
    assert candidate.orchestrator.critical_steps == 1
    assert candidate.orchestrator.depth == 0


def test_parse_generated_task_completion_accepts_result_alias_in_tool_trace():
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    env = OSINTEnvironment(EnvironmentConfig(seed=35, n_users=18, max_steps=6))
    payload = _build_valid_candidate_payload(env, cfg)
    payload["tool_trace"] = [
        {
            "tool": call["tool_name"],
            "args": dict(call["args"]),
            "result": dict(call["output"]),
        }
        for call in payload["tool_trace"]
    ]

    candidate = parse_generated_task_completion(json.dumps(payload))
    assert candidate.tool_trace
    assert all(call.output for call in candidate.tool_trace)


def test_swarm_v2_generator_reward_is_robust_to_parse_crashes():
    """Reward function must never raise: any malformed completion gets a floor reward."""
    cfg = SelfPlayTrainingConfig(pipeline_mode="swarm_v2")
    env = OSINTEnvironment(EnvironmentConfig(seed=33, n_users=14, max_steps=6))
    reward_fn = GeneratorRewardFunction(
        graph=env.graph,
        answerer_judge=DummyJudge(answer="x"),
        weights=GeneratorRewardWeights(),
        max_support_edges=cfg.swarm_v2.validation.max_support_edges,
        pipeline_mode="swarm_v2",
        swarm_v2_validation=cfg.swarm_v2.validation,
        swarm_v2_shared_context=cfg.swarm_v2.shared_context,
        parl_max_parallel_hint=cfg.swarm_v2.generator_swarm.max_agents,
    )

    garbage_orchestrator = json.dumps(
        {
            "question": "Q?",
            "answer": "y",
            "supporting_edges": [{"src": "a", "rel": "r", "dst": "y", "confidence": 1.0}],
            "tool_trace": [],
            "orchestrator": {"spawn_count": "none"},
        }
    )

    scores = reward_fn(completions=["", "{not really json", garbage_orchestrator])
    assert len(scores) == 3
    for score in scores:
        assert -1.8 <= score <= 1.2


def test_swarm_v2_dry_run_writes_new_artifacts_and_preserves_legacy_contract(tmp_path: Path):
    env_cfg = EnvironmentConfig(seed=11, n_users=14, max_steps=6)
    train_cfg = SelfPlayTrainingConfig(
        rounds=1,
        output_dir=str(tmp_path / "self_play"),
        dry_run=True,
        pipeline_mode="swarm_v2",
        generated_tasks_per_round=3,
        generator_prompts_per_round=3,
    )

    payload = run_adversarial_self_play(env_config=env_cfg, training_config=train_cfg, dry_run=True)
    assert payload["pipeline_mode"] == "swarm_v2"
    assert len(payload["rounds"]) == 1

    artifacts = payload["rounds"][0]["artifacts"]
    for key in [
        "generator_dataset",
        "answerer_dataset",
        "generated_tasks",
        "canonical_graph_candidates",
        "replay_traces",
        "validation_reports",
    ]:
        assert Path(artifacts[key]).exists()
        loaded = json.loads(Path(artifacts[key]).read_text(encoding="utf-8"))
        assert loaded is not None

    post_eval = payload["post_training_evaluation"]
    assert Path(post_eval["path"]).exists()
    assert sorted(post_eval["answerer_models"].keys()) == ["finetuned_answerer", "original_answerer"]
    assert json.loads(Path(post_eval["path"]).read_text(encoding="utf-8"))["skipped"] is True


def test_swarm_v2_fixed_canonical_mode_reuses_prompt_candidates(tmp_path: Path):
    env_cfg = EnvironmentConfig(seed=19, n_users=14, max_steps=6)
    train_cfg = SelfPlayTrainingConfig(
        rounds=1,
        output_dir=str(tmp_path / "self_play_fixed_canonical"),
        dry_run=True,
        pipeline_mode="swarm_v2",
        canonical_graph_mode="fixed",
        generated_tasks_per_round=3,
        generator_prompts_per_round=3,
    )

    payload = run_adversarial_self_play(env_config=env_cfg, training_config=train_cfg, dry_run=True)
    artifacts = payload["rounds"][0]["artifacts"]
    candidates_payload = json.loads(Path(artifacts["canonical_graph_candidates"]).read_text(encoding="utf-8"))
    generated_payload = json.loads(Path(artifacts["generated_tasks"]).read_text(encoding="utf-8"))

    expected_graphs = {
        json.dumps((item.get("canonical_graph") if isinstance(item.get("canonical_graph"), dict) else item), sort_keys=True)
        for item in candidates_payload
        if isinstance(item, dict)
    }
    assert expected_graphs

    for task in generated_payload:
        canonical_graph = ((task.get("metadata") or {}).get("canonical_graph")) or {}
        assert json.dumps(canonical_graph, sort_keys=True) in expected_graphs
