from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import requests
from requests import RequestException

from osint_env.domain.models import LLMConfig


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


class OllamaLLMClient:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", temperature: float = 0.1, timeout_seconds: int = 240):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.timeout_seconds = int(timeout_seconds)

    @staticmethod
    def _extract_tool_calls(content: str) -> list[dict[str, Any]]:
        text = str(content or "").strip()
        if not text:
            return []
        left = text.find("{")
        right = text.rfind("}")
        if left >= 0 and right > left:
            snippet = text[left : right + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
                out: list[dict[str, Any]] = []
                for item in parsed["tool_calls"]:
                    if isinstance(item, dict) and "tool_name" in item and isinstance(item.get("args", {}), dict):
                        out.append({"tool_name": str(item["tool_name"]), "args": dict(item.get("args", {}))})
                return out
        return []

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        if tools:
            payload["tools"] = tools
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = str((data.get("message") or {}).get("content", ""))
            tool_calls = self._extract_tool_calls(content)
            return LLMResponse(content=content, tool_calls=tool_calls)
        except (RequestException, ValueError):
            # Keep episode execution resilient when local model calls are transiently slow/unavailable.
            return LLMResponse(content="", tool_calls=[])


class OpenAILLMClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 256,
        timeout_seconds: int = 240,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            completion = self.client.chat.completions.create(**kwargs)
            message = completion.choices[0].message
            content = message.content if isinstance(message.content, str) else ""

            tool_calls: list[dict[str, Any]] = []
            for tc in message.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"tool_name": tc.function.name, "args": args if isinstance(args, dict) else {}})
            return LLMResponse(content=content, tool_calls=tool_calls)
        except Exception:
            return LLMResponse(content="", tool_calls=[])


def build_llm_client(config: LLMConfig | None = None) -> LLMClient:
    cfg = config or LLMConfig()
    provider = str(cfg.provider).strip().lower()
    if provider in {"", "mock", "rule", "rule_based"}:
        return RuleBasedMockLLM()
    if provider == "ollama":
        return OllamaLLMClient(
            model=cfg.model,
            base_url=cfg.ollama_base_url,
            temperature=cfg.temperature,
            timeout_seconds=cfg.timeout_seconds,
        )
    if provider == "openai":
        api_key = cfg.openai_api_key or os.getenv(cfg.openai_api_key_env, "")
        if not api_key:
            raise ValueError(
                "OpenAI provider selected but API key is missing. "
                f"Set {cfg.openai_api_key_env} or populate openai_api_key in config."
            )
        return OpenAILLMClient(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.openai_base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_seconds=cfg.timeout_seconds,
        )
    raise ValueError(f"Unsupported llm provider: {cfg.provider}")
