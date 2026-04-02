from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from osint_env.data.generator import DatasetGenerator
from osint_env.domain.models import Action, ActionType, Edge, EnvironmentConfig, Observation, TaskInstance
from osint_env.env.openenv_compat import Env
from osint_env.env.reward import (
    build_reward_model,
    compute_answer_reward,
    compute_edge_reward,
    compute_graph_f1,
)
from osint_env.memory.store import MemoryGraph, SemanticMemory
from osint_env.platforms.tools import ToolRegistry

if TYPE_CHECKING:
    from osint_env.llm.interface import LLMClient


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
    reward_components: dict[str, float] = field(default_factory=dict)


class OSINTEnvironment(Env):
    def __init__(self, config: EnvironmentConfig, llm: "LLMClient | None" = None):
        super().__init__(
            name="OSINTEnvironment",
            state_space="json-observation",
            action_space=["CALL_TOOL", "ADD_EDGE", "ANSWER"],
            episode_max_length=config.max_steps,
        )
        self.config = config
        self.generator = DatasetGenerator(config, llm=llm)
        self.graph = self.generator.build_canonical_graph()
        self.views = self.generator.build_platform_views(self.graph)
        self.tasks = self.generator.generate_tasks(self.graph, self.views, count=24)
        self.reward_model = build_reward_model(self.graph)
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

        invalid_tool_penalty = 0.0
        try:
            if tool_name == "search_memory":
                query = str(args.get("query", "")).strip()
                top_k = int(args.get("k", 5)) if str(args.get("k", "")).strip() else 5
                results = self.semantic_memory.search(query=query, k=max(1, top_k)) if query else []
                output = {"results": results, "count": len(results)}
            else:
                output = self.tools.call(tool_name, args)
        except Exception as exc:
            output = {"error": str(exc)}
            invalid_tool_penalty = -0.25
        self.state.tool_outputs.append({"tool": tool_name, "args": args, "output": output})
        self.semantic_memory.add(f"{tool_name} {args} {output}", {"tool": tool_name})
        relevance_bonus = 0.08 * self._tool_relevance(self.state.task, output)
        total = penalty + relevance_bonus + invalid_tool_penalty
        self._accumulate_reward_components(
            {
                "tool_novelty": penalty,
                "tool_relevance": relevance_bonus,
                "invalid_tool_penalty": invalid_tool_penalty,
            }
        )
        return total

    def _handle_add_edge(self, payload: dict[str, Any]) -> float:
        if self.state is None:
            return 0.0
        edge = Edge(payload["src"], payload["rel"], payload["dst"], float(payload.get("confidence", 1.0)))
        existing_edges = list(self.memory_graph.edges)
        added = self.memory_graph.add_edge(edge)
        if not added:
            self._accumulate_reward_components({"duplicate_edge_penalty": -0.15})
            return -0.15

        breakdown = compute_edge_reward(
            edge=edge,
            task=self.state.task,
            existing_edges=existing_edges,
            step_count=self.state.step_count,
            model=self.reward_model,
            graph=self.graph,
        )
        self._accumulate_reward_components(breakdown.to_dict())
        return breakdown.total

    def _handle_answer(self, payload: dict[str, Any]) -> float:
        if self.state is None:
            return 0.0
        proposed = str(payload.get("answer", "")).strip()
        self.state.answer = proposed
        self.state.done = True
        breakdown = compute_answer_reward(
            proposed_answer=proposed,
            task=self.state.task,
            pred_edges=self.memory_graph.edges,
            tool_outputs=self.state.tool_outputs,
            step_count=self.state.step_count,
            model=self.reward_model,
        )
        self._accumulate_reward_components(breakdown.to_dict())
        return breakdown.total

    def _tool_relevance(self, task: TaskInstance, output: dict[str, Any]) -> float:
        haystack = str(output).lower()
        clues = {task.answer.lower()}
        for edge in task.supporting_edges:
            clues.add(edge.src.lower())
            clues.add(edge.dst.lower())
            clues.add(edge.rel.lower())
        if not clues:
            return 0.0
        matches = sum(1 for token in clues if token in haystack)
        return matches / len(clues)

    def _accumulate_reward_components(self, values: dict[str, float]) -> None:
        if self.state is None:
            return
        for key, value in values.items():
            self.state.reward_components[key] = self.state.reward_components.get(key, 0.0) + float(value)

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
            "graph_f1": compute_graph_f1(self.memory_graph.edges, self.state.task.supporting_edges),
            "reward_components": dict(self.state.reward_components),
        }
