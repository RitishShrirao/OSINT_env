import json
import re
from threading import Lock

from osint_env.data.generator import (
    DatasetGenerator,
    build_swarm_v2_canonical_subgraph,
    build_swarm_v2_path_candidates,
    build_swarm_v2_tool_trace,
    emit_swarm_v2_question,
    select_swarm_v2_answer,
    trace_swarm_v2_path,
)
from osint_env.domain.models import EnvironmentConfig
from osint_env.llm.interface import LLMResponse


class SharedContextLLM:
    def __init__(self):
        self.prompts: list[str] = []
        self._lock = Lock()

    def generate(self, messages, tools):
        prompt = str(messages[0].get("content", "")) if messages else ""
        with self._lock:
            self.prompts.append(prompt)

        if "SEED_GRAPH_EXPANSION_AGENT" in prompt:
            worker_match = re.search(r"worker_id:\s*(\d+)", prompt)
            worker_idx = int(worker_match.group(1)) if worker_match else 0
            payload = {
                "edges": [
                    {
                        "src": "user_0",
                        "rel": f"llm_rel_{worker_idx}",
                        "dst": "user_1",
                        "confidence": 0.9,
                    }
                ]
            }
            return LLMResponse(content=json.dumps(payload), tool_calls=[])

        if "SEED_TASK_EXPANSION_AGENT" in prompt:
            worker_match = re.search(r"worker_id:\s*(\d+)", prompt)
            worker_idx = int(worker_match.group(1)) if worker_match else 0
            budget_match = re.search(r"task_budget:\s*(\d+)", prompt)
            task_budget = int(budget_match.group(1)) if budget_match else 1
            tasks = []
            for local_idx in range(max(1, task_budget)):
                tasks.append(
                    {
                        "task_type": "identity_resolution",
                        "question": f"Which canonical user is tied to alias alias_seed_{worker_idx}_{local_idx}?",
                        "answer": "user_1",
                        "supporting_edges": [
                            {
                                "src": "alias_seed_0",
                                "rel": "alias_of",
                                "dst": "user_1",
                                "confidence": 0.95,
                            }
                        ],
                    }
                )
            payload = {"tasks": tasks}
            return LLMResponse(content=json.dumps(payload), tool_calls=[])

        return LLMResponse(content="{}", tool_calls=[])


def test_generator_outputs():
    gen = DatasetGenerator(EnvironmentConfig(n_users=20, seed=11))
    graph = gen.build_canonical_graph()
    views = gen.build_platform_views(graph)
    tasks = gen.generate_tasks(graph, views, count=5)
    assert len(graph.nodes) >= 20
    assert len(views.microblog_posts) >= 20
    assert len(tasks) == 5


def test_seeded_views_include_seeded_posts_and_threads():
    from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config

    shared = load_shared_config("datasets/fixed_levels/shared_config_fixed_levels.json")
    cfg = clone_environment_config(shared.environment)
    cfg.seeding = load_seeding_config("datasets/fixed_levels/seed_fixed_levels.json")
    cfg.llm.provider = "mock"

    gen = DatasetGenerator(cfg)
    graph = gen.build_canonical_graph()
    views = gen.build_platform_views(graph)

    seeded_post = next((post for post in views.microblog_posts if post["post_id"] == "post_midnight_manifest"), None)
    seeded_thread = next((thread for thread in views.forum_threads if thread["thread_id"] == "thr_supply_leak"), None)

    assert seeded_post is not None
    assert "loc_dockyard17" in seeded_post["references"]
    assert seeded_thread is not None
    assert "org_northbridge_logistics" in seeded_thread["references"]


def test_graph_generation_uses_parallel_shared_context_workers():
    cfg = EnvironmentConfig(n_users=12, seed=9)
    cfg.seeding.llm_generate_remaining_graph = True
    cfg.seeding.llm_generated_edge_budget = 4
    cfg.seeding.llm_generate_remaining_tasks = False
    cfg.seeding.llm_generation_parallel = True
    cfg.seeding.llm_generation_workers = 3
    cfg.seeding.llm_generation_retries = 1
    cfg.seeding.allow_template_fallback_on_llm_failure = False

    llm = SharedContextLLM()
    gen = DatasetGenerator(cfg, llm=llm)
    graph = gen.build_canonical_graph()

    assert any(edge.rel.startswith("llm_rel_") for edge in graph.edges)
    graph_prompts = [prompt for prompt in llm.prompts if "SEED_GRAPH_EXPANSION_AGENT" in prompt]
    assert len(graph_prompts) >= 2
    assert all("SHARED_CONTEXT" in prompt for prompt in graph_prompts)


def test_task_generation_uses_parallel_shared_context_workers():
    cfg = EnvironmentConfig(n_users=12, seed=13)
    cfg.seeding.llm_generate_remaining_graph = False
    cfg.seeding.llm_generate_remaining_tasks = True
    cfg.seeding.llm_generated_task_budget = 4
    cfg.seeding.llm_generation_parallel = True
    cfg.seeding.llm_generation_workers = 3
    cfg.seeding.llm_generation_retries = 1
    cfg.seeding.allow_template_fallback_on_llm_failure = False

    llm = SharedContextLLM()
    gen = DatasetGenerator(cfg, llm=llm)
    graph = gen.build_canonical_graph()
    views = gen.build_platform_views(graph)
    tasks = gen.generate_tasks(graph, views, count=4)

    assert len(tasks) == 4
    assert any(task.metadata.get("shared_context") for task in tasks)
    task_prompts = [prompt for prompt in llm.prompts if "SEED_TASK_EXPANSION_AGENT" in prompt]
    assert len(task_prompts) >= 2
    assert all("SHARED_CONTEXT" in prompt for prompt in task_prompts)


def test_swarm_v2_path_tools_replay_a_valid_multi_hop_trace():
    gen = DatasetGenerator(EnvironmentConfig(n_users=20, seed=17))
    graph = gen.build_canonical_graph()
    candidates = build_swarm_v2_path_candidates(graph, gen.rng, count=4, min_hops=2, max_hops=3)

    assert candidates
    traced = trace_swarm_v2_path(graph, candidates[0])
    assert traced
    assert len(traced) >= 2

    question = emit_swarm_v2_question(traced)
    answer = select_swarm_v2_answer(traced)
    tool_trace = build_swarm_v2_tool_trace(graph, traced)
    canonical = build_swarm_v2_canonical_subgraph(graph, traced, max_extra_edges=2)

    assert question.startswith("If you start at")
    assert answer == traced[-1].dst
    assert any(call["tool_name"] == "trace_path" for call in tool_trace)
    assert canonical["path"]
    assert canonical["answer"] == answer
