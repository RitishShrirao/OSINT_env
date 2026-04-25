from osint_env.env.spawn_reward_hooks import critical_steps, parl_style_spawn_reward


def test_critical_steps_matches_parallel_path_length():
    total = critical_steps(main_steps=[1, 1, 1], parallel_subagent_steps=[[3, 2], [0], [4, 1, 2]])
    assert total == 1 + 3 + 1 + 0 + 1 + 4


def test_parl_reward_prefers_finished_parallel_work():
    base = parl_style_spawn_reward(
        task_outcome_reward=0.2,
        spawn_count=4,
        finished_subtasks=1,
        critical_steps=12,
        lambda_parallel=0.2,
        lambda_finish=0.25,
        anneal=1.0,
        breadth=2,
        depth=3,
    )
    better = parl_style_spawn_reward(
        task_outcome_reward=0.2,
        spawn_count=4,
        finished_subtasks=4,
        critical_steps=8,
        lambda_parallel=0.2,
        lambda_finish=0.25,
        anneal=1.0,
        breadth=4,
        depth=2,
    )
    assert better > base


def test_parl_auxiliary_can_be_annealed_out():
    frozen = parl_style_spawn_reward(
        task_outcome_reward=0.7,
        spawn_count=8,
        finished_subtasks=8,
        critical_steps=5,
        anneal=0.0,
    )
    assert frozen == 0.7
