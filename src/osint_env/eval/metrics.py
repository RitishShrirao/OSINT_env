from __future__ import annotations

import math
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
    total_knowledge_carrier: float = 0.0
    total_knowledge_indexing: float = 0.0
    total_connectivity: float = 0.0
    total_format_reward: float = 0.0
    total_relation_informativeness: float = 0.0
    total_entity_informativeness: float = 0.0
    total_diversity: float = 0.0
    total_soft_shaping: float = 0.0
    total_connectivity_gain: float = 0.0
    total_compactness: float = 0.0
    total_spawn_count: int = 0
    total_spawn_finished_subtasks: int = 0
    total_spawn_critical_steps: int = 0

    @staticmethod
    def _sigmoid_temperature(value: float, temperature: float = 2.0) -> float:
        scaled = float(value) / max(1e-6, float(temperature))
        if scaled >= 0:
            z = math.exp(-scaled)
            return 1.0 / (1.0 + z)
        z = math.exp(scaled)
        return z / (1.0 + z)

    def add(self, info: dict, task_type: str, graph_f1: float) -> None:
        self.episodes += 1
        ok = info.get("agent_answer") == info.get("task_answer")
        self.success += int(ok)
        self.total_steps += int(info.get("step_count", 0))
        self.total_tool_calls += int(info.get("tool_calls", 0))
        self.total_redundant_tool_calls += int(info.get("redundant_tool_calls", 0))
        self.total_reward += float(info.get("total_reward", 0.0))
        self.graph_f1_scores.append(graph_f1)
        components = info.get("reward_components", {})
        self.total_knowledge_carrier += float(components.get("knowledge_carrier", 0.0))
        self.total_knowledge_indexing += float(components.get("knowledge_indexing", 0.0))
        self.total_connectivity += float(components.get("connectivity", 0.0))
        self.total_format_reward += float(components.get("format_reward", 0.0))
        self.total_relation_informativeness += float(components.get("relation_informativeness", 0.0))
        self.total_entity_informativeness += float(components.get("entity_informativeness", 0.0))
        self.total_diversity += float(components.get("diversity", 0.0))
        self.total_soft_shaping += float(components.get("soft_shaping", 0.0))
        self.total_connectivity_gain += float(components.get("connectivity_gain", 0.0))
        self.total_compactness += float(components.get("compactness", 0.0))
        self.total_spawn_count += int(info.get("spawn_count", 0))
        self.total_spawn_finished_subtasks += int(info.get("spawn_finished_subtasks", 0))
        self.total_spawn_critical_steps += int(info.get("spawn_critical_steps", 0))
        if task_type == "identity_resolution":
            self.deanonymization_total += 1
            self.deanonymization_success += int(ok)

    def summary(self) -> dict:
        episodes = max(1, self.episodes)
        task_success_rate = self.success / episodes
        tool_efficiency = 1.0 - (self.total_redundant_tool_calls / max(1, self.total_tool_calls))
        avg_graph_f1 = sum(self.graph_f1_scores) / max(1, len(self.graph_f1_scores))
        deanonymization_accuracy = self.deanonymization_success / max(1, self.deanonymization_total)
        avg_reward_raw = self.total_reward / episodes
        avg_reward = self._sigmoid_temperature(avg_reward_raw, temperature=2.0)
        avg_knowledge_carrier = self.total_knowledge_carrier / episodes
        avg_knowledge_indexing = self.total_knowledge_indexing / episodes
        avg_connectivity = self.total_connectivity / episodes
        avg_relation_informativeness = self.total_relation_informativeness / episodes
        avg_entity_informativeness = self.total_entity_informativeness / episodes
        avg_diversity = self.total_diversity / episodes
        avg_soft_shaping = self.total_soft_shaping / episodes
        avg_connectivity_gain = self.total_connectivity_gain / episodes
        avg_compactness = self.total_compactness / episodes
        avg_spawn_count = self.total_spawn_count / episodes
        spawn_completion = self.total_spawn_finished_subtasks / max(1, self.total_spawn_count)
        avg_spawn_critical_steps = self.total_spawn_critical_steps / episodes
        spawn_latency_signal = 1.0 / max(1.0, avg_spawn_critical_steps)
        spawn_signal = max(0.0, min(1.0, 0.6 * spawn_completion + 0.4 * spawn_latency_signal))

        reward_norm = avg_reward
        retrieval_signal = max(0.0, min(1.0, 0.5 + 0.35 * avg_knowledge_carrier + 0.35 * avg_knowledge_indexing))
        structural_signal = max(
            0.0,
            min(
                1.0,
                0.5
                + 0.25 * avg_connectivity
                + 0.20 * avg_relation_informativeness
                + 0.20 * avg_entity_informativeness
                + 0.15 * avg_diversity
                + 0.10 * avg_connectivity_gain,
            ),
        )
        leaderboard_score = (
            0.28 * task_success_rate
            + 0.20 * avg_graph_f1
            + 0.12 * tool_efficiency
            + 0.12 * deanonymization_accuracy
            + 0.14 * retrieval_signal
            + 0.09 * structural_signal
            + 0.05 * reward_norm
            + 0.04 * spawn_signal
        )
        return {
            "task_success_rate": task_success_rate,
            "tool_efficiency": tool_efficiency,
            "avg_graph_f1": avg_graph_f1,
            "avg_steps_to_solution": self.total_steps / episodes,
            "deanonymization_accuracy": deanonymization_accuracy,
            "avg_reward": avg_reward,
            "avg_knowledge_carrier_reward": avg_knowledge_carrier,
            "avg_knowledge_indexing_reward": avg_knowledge_indexing,
            "avg_connectivity_reward": avg_connectivity,
            "avg_format_reward": self.total_format_reward / episodes,
            "avg_relation_informativeness_reward": avg_relation_informativeness,
            "avg_entity_informativeness_reward": avg_entity_informativeness,
            "avg_diversity_reward": avg_diversity,
            "avg_soft_shaping_reward": avg_soft_shaping,
            "avg_connectivity_gain_reward": avg_connectivity_gain,
            "avg_compactness_reward": avg_compactness,
            "avg_spawn_count": avg_spawn_count,
            "spawn_completion_rate": spawn_completion,
            "avg_spawn_critical_steps": avg_spawn_critical_steps,
            "spawn_signal": spawn_signal,
            "retrieval_signal": retrieval_signal,
            "structural_signal": structural_signal,
            "leaderboard_score": leaderboard_score,
        }
