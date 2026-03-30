from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openenv.env import Env

from osint_env.data.generator import DatasetGenerator
from osint_env.domain.models import Action, ActionType, Edge, EnvironmentConfig, Observation, TaskInstance
from osint_env.env.reward import compute_graph_f1, edge_in_truth
from osint_env.memory.store import MemoryGraph, SemanticMemory
from osint_env.platforms.tools import ToolRegistry


@dataclass(slots=True)
class EpisodeState:
    task: TaskInstance
    step_count: int = 0
    done: bool = False
    total_reward: float = 0.0
    tool_calls: int = 0
    redundant_tool_calls: int = 0
    action_history: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    call_fingerprints: set[str] = field(default_factory=set)


class OSINTEnvironment(Env):
    def __init__(self, config: EnvironmentConfig):
        super().__init__(
            name="OSINTEnvironment",
            state_space="json-observation",
            action_space=["CALL_TOOL", "ADD_EDGE", "ANSWER"],
            episode_max_length=config.max_steps,
        )
        self.config = config
        self.generator = DatasetGenerator(config)
        self.graph = self.generator.build_canonical_graph()
        self.views = self.generator.build_platform_views(self.graph)
        self.tasks = self.generator.generate_tasks(self.graph, self.views, count=24)
        self.tools = ToolRegistry(self.views)
        self.memory_graph = MemoryGraph()
        self.semantic_memory = SemanticMemory()
        self._task_idx = 0
        self.state: EpisodeState | None = None

    def reset(self) -> Observation:
        task = self.tasks[self._task_idx % len(self.tasks)]
        self._task_idx += 1
        self.state = EpisodeState(task=task)
        self.memory_graph = MemoryGraph()
        self.semantic_memory = SemanticMemory()
        return self._observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.state.done:
            return self._observation(), 0.0, True, self._info()

        self.state.step_count += 1
        reward = 0.0

        if action.action_type == ActionType.CALL_TOOL:
            reward += self._handle_tool(action.payload)
        elif action.action_type == ActionType.ADD_EDGE:
            reward += self._handle_add_edge(action.payload)
        elif action.action_type == ActionType.ANSWER:
            reward += self._handle_answer(action.payload)
        else:
            reward -= 0.5

        if self.state.step_count >= self.config.max_steps and not self.state.done:
            self.state.done = True
            reward -= 0.3

        self.state.total_reward += reward
        self.state.action_history.append({"type": action.action_type.value, "payload": action.payload, "reward": reward})
        return self._observation(), reward, self.state.done, self._info()

    def _handle_tool(self, payload: dict[str, Any]) -> float:
        if self.state is None:
            return 0.0
        tool_name = payload["tool_name"]
        args = payload.get("args", {})
        fp = f"{tool_name}:{sorted(args.items())}"
        self.state.tool_calls += 1
        if fp in self.state.call_fingerprints:
            self.state.redundant_tool_calls += 1
            penalty = -0.2
        else:
            penalty = 0.05
            self.state.call_fingerprints.add(fp)

        output = self.tools.call(tool_name, args)
        self.state.tool_outputs.append({"tool": tool_name, "args": args, "output": output})
        self.semantic_memory.add(f"{tool_name} {args} {output}", {"tool": tool_name})
        return penalty

    def _handle_add_edge(self, payload: dict[str, Any]) -> float:
        if self.state is None:
            return 0.0
        edge = Edge(payload["src"], payload["rel"], payload["dst"], float(payload.get("confidence", 1.0)))
        added = self.memory_graph.add_edge(edge)
        if not added:
            return -0.15
        return 0.3 if edge_in_truth(edge, self.state.task) else -0.25

    def _handle_answer(self, payload: dict[str, Any]) -> float:
        if self.state is None:
            return 0.0
        proposed = str(payload.get("answer", "")).strip()
        self.state.answer = proposed
        self.state.done = True
        final = 2.0 if proposed == self.state.task.answer else -1.0
        f1 = compute_graph_f1(self.memory_graph.edges, self.state.task.supporting_edges)
        return final + (0.5 * f1)

    def _observation(self) -> Observation:
        if self.state is None:
            raise RuntimeError("State is not initialized.")
        return Observation(
            tool_outputs=self.state.tool_outputs[-5:],
            graph_snapshot=self.memory_graph.to_snapshot(),
            action_history=self.state.action_history[-10:],
            task={"task_id": self.state.task.task_id, "task_type": self.state.task.task_type, "question": self.state.task.question},
        )

    def _info(self) -> dict[str, Any]:
        if self.state is None:
            return {}
        return {
            "step_count": self.state.step_count,
            "total_reward": self.state.total_reward,
            "tool_calls": self.state.tool_calls,
            "redundant_tool_calls": self.state.redundant_tool_calls,
            "task_answer": self.state.task.answer,
            "agent_answer": self.state.answer,
        }
