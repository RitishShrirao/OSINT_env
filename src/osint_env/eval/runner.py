from __future__ import annotations

from osint_env.agents.single_agent import SingleAgentRunner
from osint_env.agents.swarm_agent import SwarmAgentRunner
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.reward import compute_graph_f1
from osint_env.eval.metrics import EvalMetrics
from osint_env.llm.interface import LLMClient


def run_evaluation(
    env: OSINTEnvironment,
    episodes: int = 20,
    return_details: bool = False,
    llm: LLMClient | None = None,
) -> dict:
    metrics = EvalMetrics()
    if env.config.swarm.enabled:
        runner = SwarmAgentRunner(env=env, llm=llm)
    else:
        runner = SingleAgentRunner(env=env, llm=llm)
    episode_rows: list[dict] = []
    for _ in range(episodes):
        info = runner.run_episode()
        task_type = env.state.task.task_type if env.state else "unknown"
        task_id = env.state.task.task_id if env.state else "unknown"
        truth = env.state.task.supporting_edges if env.state else []
        pred = env.memory_graph.edges if env.state else []
        graph_f1 = compute_graph_f1(pred, truth)
        metrics.add(info, task_type=task_type, graph_f1=graph_f1)
        episode_rows.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "graph_f1": graph_f1,
                "reward": float(info.get("total_reward", 0.0)),
                "steps": int(info.get("step_count", 0)),
                "tool_calls": int(info.get("tool_calls", 0)),
                "success": int(info.get("agent_answer") == info.get("task_answer")),
                "reward_components": dict(info.get("reward_components", {})),
                "spawn_count": int(info.get("spawn_count", 0)),
                "spawn_critical_steps": int(info.get("spawn_critical_steps", 0)),
            }
        )
    summary = metrics.summary()
    if return_details:
        return {"summary": summary, "episodes": episode_rows}
    return summary
