from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests

from osint_env.baselines.openai_runner import SYSTEM_PROMPT, build_action_tools
from osint_env.llm.interface import OllamaLLMClient

SPACE_URL = os.getenv("SPACE_URL", "https://siddeshwar1625-osint.hf.space").rstrip("/")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:2b")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
TASK_INDICES = [int(x.strip()) for x in os.getenv("TASK_INDICES", "0").split(",") if x.strip()]


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


def _assistant_tool_call_id(message: dict[str, Any]) -> str | None:
    tool_calls = list(message.get("tool_calls", []))
    if not tool_calls:
        return None
    tool_call_id = tool_calls[0].get("id")
    return str(tool_call_id) if tool_call_id else None


def _tool_result_message(assistant_message: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    tool_call_id = _assistant_tool_call_id(assistant_message)
    if not tool_call_id:
        return None
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, sort_keys=True),
    }


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


def _format_action(action: dict[str, Any]) -> str:
    action_type = str(action.get("action_type", ""))
    payload = dict(action.get("payload", {}))
    if action_type == "ANSWER":
        return f"answer({payload.get('answer', 'unknown')})"
    if action_type == "ADD_EDGE":
        return (
            "add_edge("
            f"{payload.get('src', '')},"
            f"{payload.get('rel', '')},"
            f"{payload.get('dst', '')},"
            f"{float(payload.get('confidence', 1.0)):.2f}"
            ")"
        )
    tool_name = str(payload.get("tool_name", "tool"))
    args = dict(payload.get("args", {}))
    if not args:
        return f"{tool_name}()"
    arg_str = ",".join(f"{key}={value}" for key, value in sorted(args.items()))
    return f"{tool_name}({arg_str})"


def get_model_action(client: OllamaLLMClient, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    llm_resp = client.generate(messages, tools)
    content = llm_resp.content or ""
    tool_calls = list(llm_resp.tool_calls or [])
    if not tool_calls:
        return {"action_type": "ANSWER", "payload": {"answer": content.strip() or "unknown"}}, {
            "role": "assistant",
            "content": content,
        }

    tool_call = tool_calls[0]
    tool_name = str(tool_call.get("tool_name", ""))
    args = dict(tool_call.get("args", {}))
    assistant_message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "local",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args, sort_keys=True)},
            }
        ],
    }
    return _decode_action(tool_name, args), assistant_message


def main() -> None:
    try:
        ping = requests.get(f"{SPACE_URL}/healthz", timeout=REQUEST_TIMEOUT)
        ping.raise_for_status()
        print(f"Space health: {ping.json()}")
    except Exception as exc:
        raise SystemExit(f"Space health check failed: {exc}") from exc

    client = OllamaLLMClient(model=MODEL, base_url=OLLAMA_BASE, timeout_seconds=REQUEST_TIMEOUT)
    tools = build_action_tools()

    for task_index in TASK_INDICES:
        print(f"Resetting task {task_index} via {SPACE_URL}/openenv/reset")
        resp = requests.post(f"{SPACE_URL}/openenv/reset", json={"task_index": task_index}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        session_id = str(data.get("session_id"))
        observation = data.get("observation", {})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation, indent=2, sort_keys=True)},
        ]

        done = bool(data.get("done", False))
        step = 0
        rewards: list[float] = []

        while not done and step < MAX_STEPS:
            step += 1
            action, assistant_message = get_model_action(client, messages, tools)
            error = None
            try:
                result = requests.post(
                    f"{SPACE_URL}/openenv/step",
                    json={
                        "session_id": session_id,
                        "action_type": action["action_type"],
                        "payload": action["payload"],
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                result.raise_for_status()
                result = result.json()
            except Exception as exc:
                error = str(exc)
                print(f"Step {step}: request failed: {error}")
                break

            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(result.get("done", False))
            rewards.append(reward)
            print(f"Step {step}: action={_format_action(action)} reward={reward:.3f} done={done} error={error}")

            messages.append(assistant_message)
            tool_message = _tool_result_message(assistant_message, result)
            if tool_message is not None:
                messages.append(tool_message)

        print(f"Episode finished. steps={step} total_reward={sum(rewards):.3f} rewards={rewards}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(1)
