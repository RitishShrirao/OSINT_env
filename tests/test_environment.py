from osint_env.domain.models import Action, ActionType, EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment


def test_episode_flow():
    env = OSINTEnvironment(EnvironmentConfig(max_steps=5, seed=5))
    obs = env.reset()
    assert "question" in obs.task
    assert isinstance(obs.task.get("grader"), dict)
    assert "type" in obs.task["grader"]
    obs, r1, done, _ = env.step(Action(ActionType.CALL_TOOL, {"tool_name": "search_posts", "args": {"query": "Update"}}))
    assert done is False
    assert isinstance(r1, float)
    _, r2, done, info = env.step(Action(ActionType.ANSWER, {"answer": "unknown"}))
    assert done is True
    assert "total_reward" in info
    assert isinstance(r2, float)


def test_search_memory_tool_returns_results_after_tool_use():
    env = OSINTEnvironment(EnvironmentConfig(max_steps=6, seed=5))
    env.reset()
    env.step(Action(ActionType.CALL_TOOL, {"tool_name": "search_posts", "args": {"query": "Update"}}))
    obs, reward, done, _ = env.step(
        Action(ActionType.CALL_TOOL, {"tool_name": "search_memory", "args": {"query": "Update", "k": 3}})
    )
    assert done is False
    assert isinstance(reward, float)
    assert obs.tool_outputs[-1]["tool"] == "search_memory"
    assert obs.tool_outputs[-1]["output"]["count"] >= 1


def test_search_shared_context_returns_task_local_hits():
    env = OSINTEnvironment(EnvironmentConfig(max_steps=6, seed=7))
    obs = env.reset()
    assert obs.task["shared_context_available"] is True
    answer = str(env.state.task.answer if env.state else "")

    obs, reward, done, _ = env.step(
        Action(ActionType.CALL_TOOL, {"tool_name": "search_shared_context", "args": {"query": answer, "k": 5}})
    )
    assert done is False
    assert isinstance(reward, float)
    assert obs.tool_outputs[-1]["tool"] == "search_shared_context"
    assert obs.tool_outputs[-1]["output"]["shared_context_available"] is True
    assert obs.tool_outputs[-1]["output"]["count"] >= 1


def test_invalid_tool_call_does_not_crash_episode():
    env = OSINTEnvironment(EnvironmentConfig(max_steps=4, seed=8))
    env.reset()
    _, reward, done, info = env.step(Action(ActionType.CALL_TOOL, {"tool_name": "no_such_tool", "args": {}}))
    assert done is False
    assert reward < 0
    assert "invalid_tool_penalty" in info["reward_components"]
