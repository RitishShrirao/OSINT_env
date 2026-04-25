from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class Action(BaseModel):
    """Structured action payload used by OpenEnv step()."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    payload: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Backward-compatible positional form: Action(action_type, payload)
        if args:
            if len(args) != 2:
                raise TypeError("Action() accepts either keyword fields or 2 positional args")
            if "action_type" in kwargs or "payload" in kwargs:
                raise TypeError("Action() cannot mix positional and keyword fields")
            kwargs["action_type"] = args[0]
            kwargs["payload"] = args[1]
        super().__init__(**kwargs)


class Observation(BaseModel):
    """Typed observation payload returned by reset()/step()/state()."""

    model_config = ConfigDict(extra="forbid")

    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    graph_snapshot: dict[str, Any] = Field(default_factory=dict)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    task: dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """Typed reward payload for structured reward accounting."""

    model_config = ConfigDict(extra="forbid")

    value: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)


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
    llm_generation_parallel: bool = True
    llm_generation_workers: int = 3
    llm_generation_retries: int = 2
    allow_template_fallback_on_llm_failure: bool = False


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
class LLMConfig:
    provider: str = "mock"
    model: str = "qwen3:2b"
    temperature: float = 0.1
    max_tokens: int = 256
    timeout_seconds: int = 240
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_api_key: str = ""


@dataclass(slots=True)
class EnvironmentConfig:
    n_users: int = 40
    alias_density: float = 0.35
    noise_level: float = 0.15
    red_herring_rate: float = 0.1
    max_steps: int = 18
    seed: int = 7
    dataset_mode: str = "canonical"
    metaqa_root: str = "metaQA"
    metaqa_kb_path: str = ""
    metaqa_variant: str = "vanilla"
    metaqa_hops: list[str] = field(default_factory=lambda: ["1-hop", "2-hop", "3-hop"])
    metaqa_splits: list[str] = field(default_factory=lambda: ["train", "dev", "test"])
    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    spawn_reward: SpawnRewardConfig = field(default_factory=SpawnRewardConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
