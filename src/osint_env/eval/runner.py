from __future__ import annotations

from osint_env.agents.single_agent import SingleAgentRunner
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.reward import compute_graph_f1
from osint_env.eval.metrics import EvalMetrics


def run_evaluation(env: OSINTEnvironment, episodes: int = 20) -> dict:
    metrics = EvalMetrics()
    runner = SingleAgentRunner(env=env)
    for _ in range(episodes):
        info = runner.run_episode()
        task_type = env.state.task.task_type if env.state else "unknown"
        truth = env.state.task.supporting_edges if env.state else []
        pred = env.memory_graph.edges if env.state else []
        metrics.add(info, task_type=task_type, graph_f1=compute_graph_f1(pred, truth))
    return metrics.summary()
