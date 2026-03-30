from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]]


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        ...


class RuleBasedMockLLM:
    """Deterministic fallback for local testing without model dependencies."""

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        question = ""
        for m in reversed(messages):
            if m.get("role") == "system" and "question" in m.get("content", ""):
                question = m["content"]
                break
        if "alias" in question:
            return LLMResponse(
                content="Need alias lookup.",
                tool_calls=[{"tool_name": "search_posts", "args": {"query": "Update"}}, {"tool_name": "get_profile", "args": {"user_id": "user_0"}}],
            )
        return LLMResponse(content="Need profile lookup.", tool_calls=[{"tool_name": "search_people", "args": {"org": "Apex"}}])
