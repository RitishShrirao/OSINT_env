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
