from osint_env.domain.models import Edge, EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.reward import build_reward_model, compute_answer_reward, compute_edge_reward


def test_composite_edge_reward_returns_breakdown():
    env = OSINTEnvironment(EnvironmentConfig(seed=13, n_users=16, max_steps=6))
    obs = env.reset()
    task = env.state.task

    model = build_reward_model(env.graph)
    edge = task.supporting_edges[0]
    breakdown = compute_edge_reward(
        edge=edge,
        task=task,
        existing_edges=[],
        step_count=1,
        model=model,
        graph=env.graph,
    )
    assert isinstance(breakdown.total, float)
    assert breakdown.global_accuracy > 0
    assert isinstance(breakdown.connectivity_gain, float)


def test_answer_reward_uses_graph_and_tool_context():
    env = OSINTEnvironment(EnvironmentConfig(seed=21, n_users=18, max_steps=6))
    env.reset()
    task = env.state.task

    pred_edges = [Edge(task.supporting_edges[0].src, task.supporting_edges[0].rel, task.supporting_edges[0].dst)]
    tool_outputs = [{"tool": "get_profile", "output": {"result": {"user_id": task.answer}}}]

    good = compute_answer_reward(
        proposed_answer=task.answer,
        task=task,
        pred_edges=pred_edges,
        tool_outputs=tool_outputs,
        step_count=2,
    )
    bad = compute_answer_reward(
        proposed_answer="wrong",
        task=task,
        pred_edges=[],
        tool_outputs=[],
        step_count=2,
    )

    assert good.total > bad.total
    assert good.graph_f1 >= 0
    assert isinstance(good.relation_informativeness, float)
    assert isinstance(good.entity_informativeness, float)
    assert isinstance(good.repetition_penalty, float)
