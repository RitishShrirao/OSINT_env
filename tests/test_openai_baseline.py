from osint_env.baselines.openai_runner import OpenAIBaselineConfig, OpenAIBaselineRunner, build_action_tools


def test_openai_baseline_toolset_contains_answer_and_graph_actions():
    tools = build_action_tools()
    names = {tool["function"]["name"] for tool in tools}
    assert "submit_answer" in names
    assert "add_edge" in names
    assert "search_memory" in names
    assert "get_post" in names


def test_gpt5_request_kwargs_avoid_temperature_and_use_max_completion_tokens():
    runner = OpenAIBaselineRunner.__new__(OpenAIBaselineRunner)
    runner.config = OpenAIBaselineConfig(model="gpt-5-nano", max_tokens=321, temperature=0.0, seed=7)
    runner.tools = build_action_tools()
    kwargs = runner._request_kwargs(messages=[{"role": "user", "content": "hi"}], episode_index=0)
    assert kwargs["max_completion_tokens"] == 321
    assert kwargs["reasoning_effort"] == "none"
    assert "temperature" not in kwargs
