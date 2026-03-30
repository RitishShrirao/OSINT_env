from osint_env.domain.models import Action, ActionType, EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment


def test_episode_flow():
    env = OSINTEnvironment(EnvironmentConfig(max_steps=5, seed=5))
    obs = env.reset()
    assert "question" in obs.task
    obs, r1, done, _ = env.step(Action(ActionType.CALL_TOOL, {"tool_name": "search_posts", "args": {"query": "Update"}}))
    assert done is False
    assert isinstance(r1, float)
    _, r2, done, info = env.step(Action(ActionType.ANSWER, {"answer": "unknown"}))
    assert done is True
    assert "total_reward" in info
    assert isinstance(r2, float)
