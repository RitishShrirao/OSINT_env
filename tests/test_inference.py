from inference import _looks_like_placeholder_api_key


def test_placeholder_api_key_detection():
    assert _looks_like_placeholder_api_key("your_openai_api_key") is True
    assert _looks_like_placeholder_api_key("sk-your-real-openai-key") is True
    assert _looks_like_placeholder_api_key("replace-me") is True
    assert _looks_like_placeholder_api_key("sk-proj-realistic-looking-token") is False
