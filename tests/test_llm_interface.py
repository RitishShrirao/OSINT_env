import os

import pytest
import requests

from osint_env.domain.models import LLMConfig
from osint_env.llm.interface import OllamaLLMClient, RuleBasedMockLLM, build_llm_client


def test_build_llm_client_mock_default():
    client = build_llm_client(LLMConfig(provider="mock"))
    assert isinstance(client, RuleBasedMockLLM)


def test_build_llm_client_openai_requires_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        build_llm_client(LLMConfig(provider="openai", openai_api_key="", openai_api_key_env="OPENAI_API_KEY"))


def test_build_llm_client_openai_with_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", openai_api_key_env="OPENAI_API_KEY")
    # Constructing should not fail when a key is present; actual API call is not made in this test.
    client = build_llm_client(cfg)
    assert client is not None


def test_openai_key_can_come_from_config_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", openai_api_key="cfg-key")
    client = build_llm_client(cfg)
    assert client is not None


def test_ollama_client_gracefully_handles_request_failure(monkeypatch: pytest.MonkeyPatch):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("timed out")

    monkeypatch.setattr("osint_env.llm.interface.requests.post", _raise)
    client = OllamaLLMClient(model="qwen3:2b", timeout_seconds=1)
    response = client.generate([{"role": "system", "content": "ping"}], tools=[])
    assert response.content == ""
    assert response.tool_calls == []
