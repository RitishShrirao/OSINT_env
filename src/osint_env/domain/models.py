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
class EnvironmentConfig:
    n_users: int = 40
    alias_density: float = 0.35
    noise_level: float = 0.15
    red_herring_rate: float = 0.1
    max_steps: int = 18
    seed: int = 7
