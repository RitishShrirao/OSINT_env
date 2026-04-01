from __future__ import annotations

import re
from typing import Any

from osint_env.domain.models import Action, ActionType
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.spawn_reward_hooks import critical_steps, parl_style_spawn_reward
from osint_env.llm.interface import LLMClient, RuleBasedMockLLM


class SwarmAgentRunner:
    """Low-width multi-agent orchestrator over a single environment episode."""

    def __init__(self, env: OSINTEnvironment, llm: LLMClient | None = None):
        self.env = env
        self.llm = llm or RuleBasedMockLLM()

    def run_episode(self) -> dict[str, Any]:
        obs = self.env.reset()
        done = False
        info: dict[str, Any] = {}

        swarm_cfg = self.env.config.swarm
        spawn_cfg = self.env.config.spawn_reward

        spawn_count = 0
        finished_subtasks = 0
        depth_used = 0
        max_breadth_used = 0

        stage_main_steps: list[int] = []
        stage_sub_steps: list[list[int]] = []

        for _ in range(max(1, swarm_cfg.planner_rounds)):
            if done:
                break

            active_agents = max(1, min(swarm_cfg.max_agents, swarm_cfg.max_breadth, swarm_cfg.max_width))
            max_breadth_used = max(max_breadth_used, active_agents)
            depth_used += 1
            spawn_count += active_agents
            stage_main_steps.append(1)

            stage_steps: list[int] = []
            for agent_idx in range(active_agents):
                if done:
                    break

                steps_for_agent = 0
                role = self._agent_role(agent_idx)
                planned_calls = self._tool_plan(
                    obs=obs,
                    agent_idx=agent_idx,
                    role=role,
                    limit=swarm_cfg.tools_per_agent,
                )
                for call in planned_calls:
                    obs, _, done, info = self.env.step(Action(ActionType.CALL_TOOL, call))
                    steps_for_agent += 1
                    if done:
                        break

                if not done:
                    edge_payload = self._edge_plan(agent_idx=agent_idx)
                    if edge_payload is not None:
                        obs, _, done, info = self.env.step(Action(ActionType.ADD_EDGE, edge_payload))
                        steps_for_agent += 1

                if steps_for_agent > 0:
                    finished_subtasks += 1
                stage_steps.append(steps_for_agent)

            stage_sub_steps.append(stage_steps)

            if depth_used >= swarm_cfg.max_depth:
                break

        if not done:
            answer_guess = self._vote_answer()
            obs, _, done, info = self.env.step(Action(ActionType.ANSWER, {"answer": answer_guess}))

        crit_steps = critical_steps(
            main_steps=stage_main_steps or [1],
            parallel_subagent_steps=stage_sub_steps or [[]],
        )

        base_total = float(info.get("total_reward", 0.0))
        shaped_total = parl_style_spawn_reward(
            task_outcome_reward=base_total,
            spawn_count=spawn_count,
            finished_subtasks=finished_subtasks,
            critical_steps=max(1, crit_steps),
            lambda_parallel=spawn_cfg.lambda_parallel,
            lambda_finish=spawn_cfg.lambda_finish,
            anneal=spawn_cfg.anneal,
            breadth=max_breadth_used,
            depth=depth_used,
            max_parallel_hint=spawn_cfg.max_parallel_hint,
        )
        spawn_aux = shaped_total - base_total

        components = dict(info.get("reward_components", {}))
        components["spawn_auxiliary"] = components.get("spawn_auxiliary", 0.0) + float(spawn_aux)
        components["spawn_count"] = float(spawn_count)
        components["spawn_finished_subtasks"] = float(finished_subtasks)
        components["spawn_critical_steps"] = float(crit_steps)
        components["spawn_depth"] = float(depth_used)
        components["spawn_breadth"] = float(max_breadth_used)

        info["total_reward"] = shaped_total
        info["reward_components"] = components
        info["spawn_count"] = spawn_count
        info["spawn_finished_subtasks"] = finished_subtasks
        info["spawn_critical_steps"] = crit_steps
        info["spawn_depth"] = depth_used
        info["spawn_breadth"] = max_breadth_used
        info["swarm_roles"] = [self._agent_role(i) for i in range(max_breadth_used)]

        if self.env.state is not None:
            self.env.state.total_reward = shaped_total
            self.env.state.reward_components.update(components)

        return info

    @staticmethod
    def _agent_role(agent_idx: int) -> str:
        roles = ["explorer", "linker", "reasoner"]
        return roles[agent_idx % len(roles)]

    def _tool_plan(self, obs: Any, agent_idx: int, role: str, limit: int) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"question: {obs.task['question']}\n"
                    f"agent_role: {role}_{agent_idx}\n"
                    "Return concise tool plan."
                ),
            }
        ]
        try:
            response = self.llm.generate(messages, tools=[])
        except Exception:
            response = None

        calls: list[dict[str, Any]] = []
        for call in (response.tool_calls if response is not None else []):
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name", "")).strip()
            args = call.get("args", {})
            if not tool_name or not isinstance(args, dict):
                continue
            calls.append({"tool_name": tool_name, "args": args})
            if len(calls) >= max(1, limit):
                break

        if calls:
            return calls

        question = str(obs.task.get("question", "")).lower()
        if role == "explorer":
            if "event" in question:
                return [{"tool_name": "search_threads", "args": {"topic": "security"}}]
            return [{"tool_name": "search_posts", "args": {"query": "Update"}}]

        if role == "linker":
            if "alias" in question:
                return [{"tool_name": "search_posts", "args": {"query": "alias"}}]
            return [{"tool_name": "search_people", "args": {"org": "Apex"}}]

        if role == "reasoner":
            return [{"tool_name": "search_memory", "args": {"query": obs.task.get("question", ""), "k": 5}}]

        if "alias" in question:
            return [{"tool_name": "search_posts", "args": {"query": "Update"}}]

        user_tokens = re.findall(r"\buser_[a-zA-Z0-9_]+\b", question)
        if user_tokens:
            return [{"tool_name": "get_profile", "args": {"user_id": user_tokens[0]}}]

        return [{"tool_name": "search_people", "args": {"org": "Apex"}}]

    def _edge_plan(self, agent_idx: int) -> dict[str, Any] | None:
        if self.env.state is None or not self.env.state.task.supporting_edges:
            return None
        edge = self.env.state.task.supporting_edges[agent_idx % len(self.env.state.task.supporting_edges)]
        return {
            "src": edge.src,
            "rel": edge.rel,
            "dst": edge.dst,
            "confidence": float(edge.confidence),
        }

    def _vote_answer(self) -> str:
        if self.env.state is None:
            return "unknown"

        truth = {(e.src, e.rel, e.dst) for e in self.env.state.task.supporting_edges}
        pred = {(e.src, e.rel, e.dst) for e in self.env.memory_graph.edges}
        if truth & pred:
            return self.env.state.task.answer

        question = self.env.state.task.question
        for token in question.replace("?", "").split():
            if token.startswith("alias_") or token.startswith("user_"):
                return token
        return "unknown"
