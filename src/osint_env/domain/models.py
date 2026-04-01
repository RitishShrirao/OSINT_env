from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    USER = "user"
    ALIAS = "alias"
    ORG = "org"
    LOCATION = "location"
    POST = "post"
    THREAD = "thread"
    EVENT = "event"


class ActionType(str, Enum):
    CALL_TOOL = "CALL_TOOL"
    ADD_EDGE = "ADD_EDGE"
    ANSWER = "ANSWER"


@dataclass(slots=True)
class Node:
    node_id: str
    node_type: NodeType
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    src: str
    rel: str
    dst: str
    confidence: float = 1.0


@dataclass(slots=True)
class CanonicalGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    args: dict[str, Any]


@dataclass(slots=True)
class Action:
    action_type: ActionType
    payload: dict[str, Any]


@dataclass(slots=True)
class Observation:
    tool_outputs: list[dict[str, Any]]
    graph_snapshot: dict[str, Any]
    action_history: list[dict[str, Any]]
    task: dict[str, Any]


@dataclass(slots=True)
class TaskInstance:
    task_id: str
    task_type: str
    question: str
    answer: str
    supporting_edges: list[Edge]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeedNodeSpec:
    node_id: str
    node_type: NodeType | str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeedEdgeSpec:
    src: str
    rel: str
    dst: str
    confidence: float = 1.0


@dataclass(slots=True)
class SeedQuestionSpec:
    question: str
    answer: str | None = None
    task_type: str = "seeded"
    supporting_edges: list[SeedEdgeSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeedingConfig:
    seeded_nodes: list[SeedNodeSpec] = field(default_factory=list)
    seeded_edges: list[SeedEdgeSpec] = field(default_factory=list)
    seeded_questions: list[SeedQuestionSpec] = field(default_factory=list)
    llm_generate_remaining_graph: bool = True
    llm_generate_remaining_tasks: bool = True
    llm_generated_edge_budget: int = 6
    llm_generated_task_budget: int = 8


@dataclass(slots=True)
class SwarmConfig:
    enabled: bool = False
    max_agents: int = 3
    max_breadth: int = 2
    max_width: int = 2
    max_depth: int = 2
    planner_rounds: int = 2
    tools_per_agent: int = 1


@dataclass(slots=True)
class SpawnRewardConfig:
    lambda_parallel: float = 0.15
    lambda_finish: float = 0.20
    anneal: float = 1.0
    max_parallel_hint: int = 3


@dataclass(slots=True)
class EnvironmentConfig:
    n_users: int = 40
    alias_density: float = 0.35
    noise_level: float = 0.15
    red_herring_rate: float = 0.1
    max_steps: int = 18
    seed: int = 7
    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    spawn_reward: SpawnRewardConfig = field(default_factory=SpawnRewardConfig)
