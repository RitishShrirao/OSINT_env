from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalMetrics:
    episodes: int = 0
    success: int = 0
    total_steps: int = 0
    total_tool_calls: int = 0
    total_redundant_tool_calls: int = 0
    total_reward: float = 0.0
    deanonymization_total: int = 0
    deanonymization_success: int = 0
    graph_f1_scores: list[float] = field(default_factory=list)

    def add(self, info: dict, task_type: str, graph_f1: float) -> None:
        self.episodes += 1
        ok = info.get("agent_answer") == info.get("task_answer")
        self.success += int(ok)
        self.total_steps += int(info.get("step_count", 0))
        self.total_tool_calls += int(info.get("tool_calls", 0))
        self.total_redundant_tool_calls += int(info.get("redundant_tool_calls", 0))
        self.total_reward += float(info.get("total_reward", 0.0))
        self.graph_f1_scores.append(graph_f1)
        if task_type == "identity_resolution":
            self.deanonymization_total += 1
            self.deanonymization_success += int(ok)

    def summary(self) -> dict:
        episodes = max(1, self.episodes)
        return {
            "task_success_rate": self.success / episodes,
            "tool_efficiency": 1.0 - (self.total_redundant_tool_calls / max(1, self.total_tool_calls)),
            "avg_graph_f1": sum(self.graph_f1_scores) / max(1, len(self.graph_f1_scores)),
            "avg_steps_to_solution": self.total_steps / episodes,
            "deanonymization_accuracy": self.deanonymization_success / max(1, self.deanonymization_total),
            "avg_reward": self.total_reward / episodes,
        }
