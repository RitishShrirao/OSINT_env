from osint_env.llm.interface import LLMResponse
from osint_env.agents.swarm_agent import SwarmAgentRunner
from osint_env.domain.models import EnvironmentConfig, SwarmConfig
from osint_env.env.environment import OSINTEnvironment


def test_swarm_runner_emits_spawn_telemetry():
    config = EnvironmentConfig(
        seed=14,
        max_steps=8,
        swarm=SwarmConfig(enabled=True, max_agents=3, max_breadth=2, max_width=2, max_depth=2, planner_rounds=2),
    )
    env = OSINTEnvironment(config)
    info = SwarmAgentRunner(env).run_episode()

    assert info["spawn_count"] > 0
    assert "spawn_auxiliary" in info["reward_components"]
    assert info["spawn_critical_steps"] > 0


class RecordingLLM:
    def __init__(self):
        self.tool_names: list[str] = []

    def generate(self, messages, tools):
        del messages
        self.tool_names = [tool["function"]["name"] for tool in tools]
        return LLMResponse(content="{}", tool_calls=[])


def test_swarm_runner_passes_lookup_tools_to_llm():
    config = EnvironmentConfig(
        seed=16,
        max_steps=6,
        swarm=SwarmConfig(enabled=True, max_agents=2, max_breadth=2, max_width=2, max_depth=1, planner_rounds=1),
    )
    env = OSINTEnvironment(config)
    llm = RecordingLLM()
    SwarmAgentRunner(env, llm=llm).run_episode()

    assert "search_memory" in llm.tool_names
    assert "search_shared_context" in llm.tool_names
