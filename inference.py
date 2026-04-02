from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI
from requests import RequestException

from osint_env.baselines.openai_runner import SYSTEM_PROMPT, build_action_tools


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPACE_URL = os.getenv("SPACE_URL", "https://siddeshwar1625-osint.hf.space").rstrip("/")

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
TASK_INDICES = [int(part.strip()) for part in os.getenv("TASK_INDICES", "0,10,20").split(",") if part.strip()]
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.67"))

BENCHMARK = "osint-openenv"
TASK_NAME = "fixed_levels_easy_mid_hard"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    action_text = json.dumps(action, sort_keys=True, separators=(",", ":"))
    error_text = "null" if error is None else json.dumps(error)
    print(
        f"[STEP] step={step} action={action_text} reward={reward:.4f} done={str(bool(done)).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = json.dumps([round(value, 4) for value in rewards], separators=(",", ":"))
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.4f} rewards={rewards_text}",
        flush=True,
    )


def _looks_like_placeholder_api_key(value: str) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return True
    placeholder_markers = [
        "your_openai_api_key",
        "your-key",
        "your_key",
        "your real",
        "real-openai-key",
        "replace-me",
        "changeme",
        "example",
        "<api-key>",
    ]
    if token.startswith("your_") or token.startswith("sk-your-"):
        return True
    return any(marker in token for marker in placeholder_markers)


def _supports_reasoning_effort_in_chat_completions(model: str) -> bool:
    model_name = str(model).strip().lower()
    if model_name.startswith("gpt-5.4-mini"):
        return False
    return model_name.startswith("gpt-5")


def _request_kwargs(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    if MODEL_NAME.strip().lower().startswith("gpt-5"):
        kwargs["max_completion_tokens"] = MAX_TOKENS
        if _supports_reasoning_effort_in_chat_completions(MODEL_NAME):
            kwargs["reasoning_effort"] = "none"
    else:
        kwargs["temperature"] = TEMPERATURE
        kwargs["max_tokens"] = MAX_TOKENS
    return kwargs


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _space_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{SPACE_URL}{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _space_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{SPACE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _decode_action(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "submit_answer":
        return {"action_type": "ANSWER", "payload": {"answer": str(args.get("answer", "")).strip()}}
    if tool_name == "add_edge":
        return {
            "action_type": "ADD_EDGE",
            "payload": {
                "src": str(args.get("src", "")).strip(),
                "rel": str(args.get("rel", "")).strip(),
                "dst": str(args.get("dst", "")).strip(),
                "confidence": float(args.get("confidence", 1.0)),
            },
        }
    return {"action_type": "CALL_TOOL", "payload": {"tool_name": tool_name, "args": dict(args)}}


def get_model_action(client: OpenAI, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        completion = client.chat.completions.create(**_request_kwargs(messages, tools))
        message = completion.choices[0].message
        tool_calls = list(message.tool_calls or [])
        if not tool_calls:
            fallback_answer = _message_text(message).strip() or "unknown"
            return {"action_type": "ANSWER", "payload": {"answer": fallback_answer}}, {
                "role": "assistant",
                "content": _message_text(message),
            }
        tool_call = tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        assistant_message = {
            "role": "assistant",
            "content": _message_text(message),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": str(tool_call.function.name),
                        "arguments": json.dumps(args, sort_keys=True),
                    },
                }
            ],
        }
        return _decode_action(str(tool_call.function.name), args), assistant_message
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "ANSWER", "payload": {"answer": "unknown"}}, {"role": "assistant", "content": ""}


def main() -> None:
    api_key = OPENAI_API_KEY or HF_TOKEN
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or HF_TOKEN before running inference.py.")
    if _looks_like_placeholder_api_key(api_key):
        raise SystemExit("Replace the placeholder with your real OpenAI API key.")

    try:
        ping = _space_get("/healthz")
        if ping.get("status") != "ok":
            raise SystemExit(f"Unexpected healthz payload: {ping}")
    except RequestException as exc:
        raise SystemExit(f"Space ping failed: {exc}") from exc

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=REQUEST_TIMEOUT)
    tools = build_action_tools()

    history: list[str] = []
    rewards: list[float] = []
    task_scores: list[float] = []
    steps_taken = 0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    for task_index in TASK_INDICES:
        result = _space_post("/openenv/reset", {"task_index": task_index})
        session_id = str(result["session_id"])
        done = bool(result.get("done", False))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(result["observation"], indent=2, sort_keys=True),
            },
        ]

        for local_step in range(1, MAX_STEPS + 1):
            if done:
                break
            action, assistant_message = get_model_action(client, messages, tools)
            error = None
            try:
                result = _space_post(
                    "/openenv/step",
                    {
                        "session_id": session_id,
                        "action_type": action["action_type"],
                        "payload": action["payload"],
                    },
                )
            except RequestException as exc:
                error = str(exc)
                result = _space_get(f"/openenv/state/{session_id}")
            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(result.get("done", False))
            rewards.append(reward)
            steps_taken += 1
            log_step(step=steps_taken, action=action, reward=reward, done=done, error=error)
            history.append(f"step={steps_taken} task_index={task_index} reward={reward:+.4f}")
            messages.append(assistant_message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "remote_step",
                    "content": json.dumps(result, sort_keys=True),
                }
            )
            if done:
                break

        info = dict(result.get("info", {}))
        task_answer = str(info.get("task_answer", ""))
        agent_answer = str(info.get("agent_answer", ""))
        task_scores.append(1.0 if agent_answer and agent_answer == task_answer else 0.0)

    score = sum(task_scores) / max(1, len(task_scores))
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
