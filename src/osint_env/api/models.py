from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OpenEnvTaskSummary(BaseModel):
    task_id: str
    task_type: str
    question: str
    difficulty: str = "unknown"


class OpenEnvObservationModel(BaseModel):
    tool_outputs: list[dict[str, Any]]
    graph_snapshot: dict[str, Any]
    action_history: list[dict[str, Any]]
    task: dict[str, Any]


class OpenEnvResetRequest(BaseModel):
    task_id: str | None = None
    task_index: int | None = None


class OpenEnvActionRequest(BaseModel):
    session_id: str
    action_type: str = Field(description="One of CALL_TOOL, ADD_EDGE, ANSWER.")
    payload: dict[str, Any] = Field(default_factory=dict)


class OpenEnvResponseEnvelope(BaseModel):
    session_id: str
    observation: OpenEnvObservationModel
    reward: float
    done: bool
    info: dict[str, Any]
