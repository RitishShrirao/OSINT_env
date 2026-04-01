from osint_env.domain.models import EnvironmentConfig, SwarmConfig
from osint_env.env.environment import OSINTEnvironment
from osint_env.eval.runner import run_evaluation


def test_eval_runner():
    env = OSINTEnvironment(EnvironmentConfig(seed=17))
    result = run_evaluation(env, episodes=3)
    assert "task_success_rate" in result
    assert "deanonymization_accuracy" in result
    assert "leaderboard_score" in result
    assert "avg_knowledge_indexing_reward" in result


def test_eval_runner_swarm_mode():
    env = OSINTEnvironment(
        EnvironmentConfig(seed=17, swarm=SwarmConfig(enabled=True, max_agents=3, max_breadth=2, max_width=2, max_depth=2))
    )
    result = run_evaluation(env, episodes=2)
    assert "spawn_signal" in result
    assert "avg_spawn_count" in result


def test_eval_runner_details_include_episode_answers():
    env = OSINTEnvironment(EnvironmentConfig(seed=17))
    result = run_evaluation(env, episodes=2, return_details=True)
    assert "episodes" in result
    assert len(result["episodes"]) == 2

    row = result["episodes"][0]
    assert "question" in row
    assert "task_answer" in row
    assert "agent_answer" in row
