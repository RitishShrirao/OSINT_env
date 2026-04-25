from osint_env.domain.models import (
    EnvironmentConfig,
    NodeType,
    SeedEdgeSpec,
    SeedNodeSpec,
    SeedQuestionSpec,
    SeedingConfig,
)
from osint_env.env.environment import OSINTEnvironment


def test_environment_includes_seeded_graph_and_questions():
    seeding = SeedingConfig(
        seeded_nodes=[
            SeedNodeSpec(node_id="alias_seed_001", node_type=NodeType.ALIAS, attrs={"handle": "@seed001"}),
            SeedNodeSpec(
                node_id="user_seed_001",
                node_type=NodeType.USER,
                attrs={"name": "Seed User", "org": "Helios Labs", "location": "Pune"},
            ),
        ],
        seeded_edges=[SeedEdgeSpec(src="alias_seed_001", rel="alias_of", dst="user_seed_001")],
        seeded_questions=[
            SeedQuestionSpec(
                question="Which canonical user owns alias alias_seed_001?",
                answer="user_seed_001",
                task_type="identity_resolution",
                supporting_edges=[SeedEdgeSpec(src="alias_seed_001", rel="alias_of", dst="user_seed_001")],
            )
        ],
        llm_generate_remaining_graph=False,
        llm_generate_remaining_tasks=False,
        llm_generated_edge_budget=0,
        llm_generated_task_budget=0,
    )
    env = OSINTEnvironment(EnvironmentConfig(seed=33, n_users=12, seeding=seeding))

    assert "alias_seed_001" in env.graph.nodes
    assert any(edge.src == "alias_seed_001" and edge.rel == "alias_of" and edge.dst == "user_seed_001" for edge in env.graph.edges)
    assert any("alias_seed_001" in task.question for task in env.tasks)
