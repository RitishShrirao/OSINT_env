from inference import _looks_like_placeholder_api_key, _tool_result_message


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
