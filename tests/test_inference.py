import inference
from inference import _format_action, _looks_like_placeholder_api_key, _tool_result_message
from osint_env.domain.models import EnvironmentConfig


def test_placeholder_api_key_detection():
    assert _looks_like_placeholder_api_key("your_openai_api_key") is True
    assert _looks_like_placeholder_api_key("sk-your-real-openai-key") is True
    assert _looks_like_placeholder_api_key("replace-me") is True
    assert _looks_like_placeholder_api_key("sk-proj-realistic-looking-token") is False


def test_tool_result_message_reuses_assistant_tool_call_id():
    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_post", "arguments": "{\"post_id\":\"post_midnight_manifest\"}"},
            }
        ],
    }
    result = {"reward": 0.1, "done": False}
    tool_message = _tool_result_message(assistant_message, result)
    assert tool_message is not None
    assert tool_message["tool_call_id"] == "call_123"
    assert tool_message["role"] == "tool"


def test_action_formatter_matches_single_line_style():
    assert _format_action({"action_type": "ANSWER", "payload": {"answer": "user_bharat"}}) == "answer(user_bharat)"
    assert _format_action(
        {
            "action_type": "CALL_TOOL",
            "payload": {"tool_name": "get_post", "args": {"post_id": "post_midnight_manifest"}},
        }
    ) == "get_post(post_id=post_midnight_manifest)"


def test_resolve_environment_config_applies_metaqa_overrides(monkeypatch):
    base_cfg = EnvironmentConfig()
    base_cfg.dataset_mode = "canonical"

    monkeypatch.setattr(inference, "load_shared_config", lambda _path: type("S", (), {"environment": base_cfg})())
    monkeypatch.setattr(inference, "clone_environment_config", lambda cfg: cfg)
    monkeypatch.setattr(inference, "SEED_FILE", "")

    monkeypatch.setattr(inference, "DATASET_MODE", "metaqa")
    monkeypatch.setattr(inference, "METAQA_ROOT", "metaQA")
    monkeypatch.setattr(inference, "METAQA_KB_PATH", "metaQA/kb.txt")
    monkeypatch.setattr(inference, "METAQA_VARIANT", "vanilla")
    monkeypatch.setattr(inference, "METAQA_HOPS_RAW", "1-hop,2-hop,3-hop")
    monkeypatch.setattr(inference, "METAQA_SPLITS_RAW", "train")

    monkeypatch.setattr(inference, "HF_TOKEN", "token")
    monkeypatch.setattr(inference, "API_KEY", "")
    monkeypatch.setattr(inference, "OPENAI_API_KEY", "")
    monkeypatch.setattr(inference, "OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.setattr(inference, "API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(inference, "OPENAI_BASE_URL", "")
    monkeypatch.setattr(inference, "HF_SPACE_URL", "")
    monkeypatch.setattr(inference, "MODEL_NAME", "gpt-5.4")
    monkeypatch.setattr(inference, "LLM_TIMEOUT_SECONDS", 0)

    cfg = inference._resolve_environment_config()

    assert cfg.dataset_mode == "metaqa"
    assert cfg.metaqa_root == "metaQA"
    assert cfg.metaqa_kb_path == "metaQA/kb.txt"
    assert cfg.metaqa_variant == "vanilla"
    assert cfg.metaqa_hops == ["1-hop", "2-hop", "3-hop"]
    assert cfg.metaqa_splits == ["train"]
