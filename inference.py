from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI
from requests import RequestException

from osint_env.baselines.openai_runner import SYSTEM_PROMPT, build_action_tools
from osint_env.eval.metrics import EvalMetrics


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = os.getenv("API_KEY", "")
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


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(bool(done)).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.3f} rewards={rewards_text}",
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


def _episode_row(result: dict[str, Any], task_meta: dict[str, Any]) -> dict[str, Any]:
    info = dict(result.get("info", {}))
    graph_snapshot = dict((result.get("observation") or {}).get("graph_snapshot", {}))
    task_type = str(task_meta.get("task_type", "unknown"))
    task_id = str(task_meta.get("task_id", "unknown"))
    question = str(task_meta.get("question", ""))
    task_answer = str(info.get("task_answer", ""))
    agent_answer = str(info.get("agent_answer", ""))
    graph_f1 = float(info.get("graph_f1", 0.0) or 0.0)
    return {
        "task_id": task_id,
        "task_type": task_type,
        "question": question,
        "task_answer": task_answer,
        "agent_answer": agent_answer,
        "graph_f1": graph_f1,
        "reward": float(info.get("total_reward", 0.0) or 0.0),
        "steps": int(info.get("step_count", 0) or 0),
        "tool_calls": int(info.get("tool_calls", 0) or 0),
        "success": int(bool(agent_answer) and agent_answer == task_answer),
        "reward_components": dict(info.get("reward_components", {})),
        "pred_edges": list(graph_snapshot.get("edges", [])),
        "truth_edges": [],
    }


def _publish_inference_report(summary: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    payload = {
        "run": {
            "name": "inference_py_run",
            "model": MODEL_NAME,
            "space_url": SPACE_URL,
            "task_indices": TASK_INDICES,
            "max_steps": MAX_STEPS,
        },
        "summary": summary,
        "episodes": episodes,
    }
    try:
        _space_post("/openenv/report_inference", payload)
    except RequestException as exc:
        print(f"[DEBUG] Failed to publish inference report: {exc}", flush=True)


def main() -> None:
    api_key = OPENAI_API_KEY or HF_TOKEN or API_KEY
    if not api_key:
        raise SystemExit("Set HF_TOKEN, OPENAI_API_KEY, or API_KEY before running inference.py.")
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
    episode_rows: list[dict[str, Any]] = []
    metrics = EvalMetrics()
    steps_taken = 0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    for task_index in TASK_INDICES:
        result = _space_post("/openenv/reset", {"task_index": task_index})
        session_id = str(result["session_id"])
        done = bool(result.get("done", False))
        task_meta = dict((result.get("observation") or {}).get("task", {}))
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
            log_step(step=steps_taken, action=_format_action(action), reward=reward, done=done, error=error)
            history.append(f"step={steps_taken} task_index={task_index} reward={reward:+.4f}")
            messages.append(assistant_message)
            tool_message = _tool_result_message(assistant_message, result)
            if tool_message is not None:
                messages.append(tool_message)
            if done:
                break

        info = dict(result.get("info", {}))
        task_answer = str(info.get("task_answer", ""))
        agent_answer = str(info.get("agent_answer", ""))
        task_scores.append(1.0 if agent_answer and agent_answer == task_answer else 0.0)
        episode_row = _episode_row(result, task_meta)
        episode_rows.append(episode_row)
        metrics.add(info, task_type=episode_row["task_type"], graph_f1=float(episode_row["graph_f1"]))

    score = sum(task_scores) / max(1, len(task_scores))
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    _publish_inference_report(metrics.summary(), episode_rows)


if __name__ == "__main__":
    main()
