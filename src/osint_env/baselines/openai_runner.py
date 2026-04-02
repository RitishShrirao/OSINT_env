from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.domain.models import Action, ActionType, Edge
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.reward import compute_graph_f1
from osint_env.eval.leaderboard import append_leaderboard_record, load_leaderboard
from osint_env.eval.metrics import EvalMetrics
from osint_env.viz import export_dashboard


SYSTEM_PROMPT = """You are an OSINT benchmark agent operating in a synthetic OpenEnv task.

Available actions are provided as function tools. On every turn, call exactly one tool.

Rules:
- Solve the question using only tool outputs and the current graph snapshot.
- When you have enough evidence, call submit_answer with the exact node id string.
- Use add_edge only for relationships strongly supported by the evidence you have already collected.
- Prefer concise, high-signal tool queries.
- Never guess free-form prose when a node id answer is required.
"""


@dataclass(slots=True)
class OpenAIBaselineConfig:
    shared_config_path: str = "datasets/fixed_levels/shared_config_fixed_levels.json"
    seed_file: str = "datasets/fixed_levels/seed_fixed_levels.json"
    output_path: str = "artifacts/baselines/openai_fixed_levels_latest.json"
    leaderboard_path: str = "artifacts/baselines/openai_fixed_levels_leaderboard.json"
    dashboard_path: str = "artifacts/baselines/openai_fixed_levels_dashboard.html"
    run_name: str = "openai_fixed_levels_baseline"
    model: str = "gpt-5-nano"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: int = 60
    episodes: int = 30
    max_steps: int = 8
    seed: int | None = 7
    append_leaderboard: bool = True


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def build_action_tools() -> list[dict[str, Any]]:
    return [
        _tool_schema(
            "search_posts",
            "Search microblog posts by substring query.",
            {"query": {"type": "string", "description": "Substring to search for in post text."}},
            ["query"],
        ),
        _tool_schema(
            "get_user_posts",
            "Fetch posts authored by a user or alias id.",
            {"user_id": {"type": "string", "description": "User or alias node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "get_mentions",
            "Fetch posts that mention a given canonical user id.",
            {"user_id": {"type": "string", "description": "Canonical user node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "search_threads",
            "Search forum threads by exact topic name.",
            {"topic": {"type": "string", "description": "Thread topic such as security or ai."}},
            ["topic"],
        ),
        _tool_schema(
            "get_thread",
            "Fetch a specific forum thread by id.",
            {"thread_id": {"type": "string", "description": "Thread node id."}},
            ["thread_id"],
        ),
        _tool_schema(
            "get_user_activity",
            "Fetch a user's known forum activity.",
            {"user_id": {"type": "string", "description": "Canonical user node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "get_profile",
            "Fetch a profile record by canonical user id.",
            {"user_id": {"type": "string", "description": "Canonical user node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "search_people",
            "Search profiles by name and or organization.",
            {
                "name": {"type": "string", "description": "Optional name substring.", "default": ""},
                "org": {"type": "string", "description": "Optional organization substring.", "default": ""},
            },
            [],
        ),
        _tool_schema(
            "get_connections",
            "Fetch explicit profile connections for a user.",
            {"user_id": {"type": "string", "description": "Canonical user node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "search_memory",
            "Search semantic memory over prior observations and tool outputs.",
            {
                "query": {"type": "string", "description": "Memory retrieval query."},
                "k": {"type": "integer", "description": "Top-k matches.", "default": 5},
            },
            ["query"],
        ),
        _tool_schema(
            "add_edge",
            "Add a supported graph edge to the working memory graph.",
            {
                "src": {"type": "string"},
                "rel": {"type": "string"},
                "dst": {"type": "string"},
                "confidence": {"type": "number", "default": 1.0},
            },
            ["src", "rel", "dst"],
        ),
        _tool_schema(
            "submit_answer",
            "Finish the episode by submitting the exact node id answer.",
            {"answer": {"type": "string", "description": "Exact node id answer for the task."}},
            ["answer"],
        ),
    ]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _safe_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_count": int(info.get("step_count", 0)),
        "total_reward": float(info.get("total_reward", 0.0)),
        "tool_calls": int(info.get("tool_calls", 0)),
        "redundant_tool_calls": int(info.get("redundant_tool_calls", 0)),
        "reward_components": dict(info.get("reward_components", {})),
    }


def _observation_payload(env: OSINTEnvironment, observation: Any, step_limit: int) -> dict[str, Any]:
    task = dict(observation.task)
    return {
        "task": {
            "task_id": task.get("task_id", ""),
            "task_type": task.get("task_type", ""),
            "question": task.get("question", ""),
        },
        "remaining_steps": max(0, step_limit - int(env.state.step_count if env.state else 0)),
        "recent_tool_outputs": list(observation.tool_outputs),
        "graph_snapshot": dict(observation.graph_snapshot),
        "recent_action_history": list(observation.action_history),
    }


class OpenAIBaselineRunner:
    def __init__(self, config: OpenAIBaselineConfig):
        self.config = config

        from openai import OpenAI

        if not config.api_key:
            raise ValueError(
                "OpenAI baseline requires an API key. "
                f"Set {config.api_key_env} or pass --openai-api-key."
            )

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )
        self.tools = build_action_tools()

    @staticmethod
    def _is_gpt5_family(model: str) -> bool:
        return str(model).strip().lower().startswith("gpt-5")

    def _request_kwargs(self, messages: list[dict[str, Any]], episode_index: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = int(self.config.seed) + episode_index

        if self._is_gpt5_family(self.config.model):
            # GPT-5 family chat-completions compatibility:
            # use max_completion_tokens and avoid temperature for older GPT-5 models.
            kwargs["reasoning_effort"] = "none"
        else:
            kwargs["temperature"] = self.config.temperature

        return kwargs

    def _build_environment(self) -> OSINTEnvironment:
        shared = load_shared_config(self.config.shared_config_path)
        env_cfg = clone_environment_config(shared.environment)
        env_cfg.seeding = load_seeding_config(self.config.seed_file)
        env_cfg.llm.provider = "mock"
        env_cfg.llm.model = self.config.model
        env_cfg.llm.temperature = self.config.temperature
        env_cfg.llm.max_tokens = self.config.max_tokens
        env_cfg.max_steps = min(int(env_cfg.max_steps), int(self.config.max_steps))
        return OSINTEnvironment(env_cfg)

    def _execute_action(
        self,
        env: OSINTEnvironment,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[Any, float, bool, dict[str, Any], dict[str, Any]]:
        if tool_name == "submit_answer":
            answer = str(args.get("answer", "")).strip()
            obs, reward, done, info = env.step(Action(ActionType.ANSWER, {"answer": answer}))
            result = {"submitted_answer": answer}
            return obs, reward, done, info, result

        if tool_name == "add_edge":
            payload = {
                "src": str(args.get("src", "")).strip(),
                "rel": str(args.get("rel", "")).strip(),
                "dst": str(args.get("dst", "")).strip(),
                "confidence": float(args.get("confidence", 1.0)),
            }
            obs, reward, done, info = env.step(Action(ActionType.ADD_EDGE, payload))
            return obs, reward, done, info, payload

        payload = {"tool_name": tool_name, "args": dict(args)}
        obs, reward, done, info = env.step(Action(ActionType.CALL_TOOL, payload))
        result = obs.tool_outputs[-1]["output"] if obs.tool_outputs else {}
        return obs, reward, done, info, result

    def _episode(self, env: OSINTEnvironment, episode_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = env.reset()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(_observation_payload(env, obs, env.config.max_steps), indent=2, sort_keys=True),
            },
        ]

        turn_trace: list[dict[str, Any]] = []
        raw_fingerprints: list[str] = []
        info: dict[str, Any] = {}
        done = False
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while not done and env.state is not None and env.state.step_count < env.config.max_steps:
            completion = self.client.chat.completions.create(**self._request_kwargs(messages, episode_index))
            if getattr(completion, "system_fingerprint", None):
                raw_fingerprints.append(str(completion.system_fingerprint))
            if getattr(completion, "usage", None) is not None:
                usage_totals["prompt_tokens"] += int(getattr(completion.usage, "prompt_tokens", 0) or 0)
                usage_totals["completion_tokens"] += int(getattr(completion.usage, "completion_tokens", 0) or 0)
                usage_totals["total_tokens"] += int(getattr(completion.usage, "total_tokens", 0) or 0)

            message = completion.choices[0].message
            content = _message_text(message)
            tool_calls = list(message.tool_calls or [])
            if not tool_calls:
                fallback_answer = content.strip() or "unknown"
                obs, reward, done, info = env.step(Action(ActionType.ANSWER, {"answer": fallback_answer}))
                tool_result = {
                    "submitted_answer": fallback_answer,
                    "reward": reward,
                    "done": done,
                    "observation": _observation_payload(env, obs, env.config.max_steps),
                    "info": _safe_info(info),
                }
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "tool", "tool_call_id": "fallback_submit", "content": json.dumps(tool_result)})
                turn_trace.append({"assistant_content": content, "tool_name": "submit_answer", "args": {"answer": fallback_answer}})
                break

            tool_call = tool_calls[0]
            tool_name = str(tool_call.function.name)
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            obs, reward, done, info, result = self._execute_action(env, tool_name, args)
            tool_payload = {
                "tool_name": tool_name,
                "args": args,
                "result": result,
                "reward": reward,
                "done": done,
                "observation": _observation_payload(env, obs, env.config.max_steps),
                "info": _safe_info(info),
            }
            assistant_message = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, sort_keys=True),
                        },
                    }
                ],
            }
            messages.append(assistant_message)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_payload, sort_keys=True)})
            turn_trace.append({"assistant_content": content, "tool_name": tool_name, "args": args, "reward": reward, "done": done})

        if not done:
            obs, _, done, info = env.step(Action(ActionType.ANSWER, {"answer": "unknown"}))
            turn_trace.append({"assistant_content": "", "tool_name": "submit_answer", "args": {"answer": "unknown"}, "reward": 0.0, "done": done})

        info = dict(info)
        info["openai_system_fingerprints"] = raw_fingerprints
        info["usage"] = usage_totals
        return info, {"turns": turn_trace}

    def run(self) -> dict[str, Any]:
        env = self._build_environment()
        metrics = EvalMetrics()
        episode_rows: list[dict[str, Any]] = []

        started = perf_counter()
        run_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for episode_index in range(int(self.config.episodes)):
            info, trace = self._episode(env, episode_index)
            episode_usage = dict(info.get("usage", {}))
            for key in run_usage:
                run_usage[key] += int(episode_usage.get(key, 0) or 0)
            task_type = env.state.task.task_type if env.state else "unknown"
            task_id = env.state.task.task_id if env.state else f"episode_{episode_index}"
            truth = env.state.task.supporting_edges if env.state else []
            pred = env.memory_graph.edges if env.state else []
            graph_f1 = compute_graph_f1(pred, truth)
            metrics.add(info, task_type=task_type, graph_f1=graph_f1)
            episode_rows.append(
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "question": env.state.task.question if env.state else "",
                    "task_answer": str(info.get("task_answer", "")),
                    "agent_answer": str(info.get("agent_answer", "")) if info.get("agent_answer") is not None else "",
                    "graph_f1": graph_f1,
                    "reward": float(info.get("total_reward", 0.0)),
                    "steps": int(info.get("step_count", 0)),
                    "tool_calls": int(info.get("tool_calls", 0)),
                    "success": int(info.get("agent_answer") == info.get("task_answer")),
                    "reward_components": dict(info.get("reward_components", {})),
                    "pred_edges": [
                        {
                            "src": edge.src,
                            "rel": edge.rel,
                            "dst": edge.dst,
                            "confidence": float(edge.confidence),
                        }
                        for edge in pred
                    ],
                    "truth_edges": [
                        {
                            "src": edge.src,
                            "rel": edge.rel,
                            "dst": edge.dst,
                            "confidence": float(edge.confidence),
                        }
                        for edge in truth
                    ],
                    "trace": trace,
                    "openai_system_fingerprints": list(info.get("openai_system_fingerprints", [])),
                    "usage": episode_usage,
                }
            )

        summary = metrics.summary()
        duration_seconds = perf_counter() - started
        if self.config.append_leaderboard:
            record = append_leaderboard_record(
                path=self.config.leaderboard_path,
                summary=summary,
                episodes=int(self.config.episodes),
                run_name=self.config.run_name,
                config={
                    "provider": "openai",
                    "model": self.config.model,
                    "seed": self.config.seed,
                    "max_steps": self.config.max_steps,
                    "shared_config_path": self.config.shared_config_path,
                    "seed_file": self.config.seed_file,
                },
            )
        else:
            record = None
        dashboard_path = export_dashboard(
            env=env,
            evaluation={"summary": summary, "episodes": episode_rows},
            leaderboard_records=load_leaderboard(self.config.leaderboard_path),
            output_path=self.config.dashboard_path,
        )

        payload: dict[str, Any] = {
            "run": {
                "name": self.config.run_name,
                "model": self.config.model,
                "episodes": int(self.config.episodes),
                "temperature": float(self.config.temperature),
                "max_tokens": int(self.config.max_tokens),
                "timeout_seconds": int(self.config.timeout_seconds),
                "max_steps": int(self.config.max_steps),
                "seed": self.config.seed,
                "shared_config_path": self.config.shared_config_path,
                "seed_file": self.config.seed_file,
                "duration_seconds": duration_seconds,
                "dashboard_path": dashboard_path,
            },
            "summary": summary,
            "usage": run_usage,
            "episodes": episode_rows,
        }

        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if record is not None:
            payload["record"] = record
            output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        return payload
