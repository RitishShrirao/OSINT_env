from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from server import app
from osint_env.baselines.openai_runner import OpenAIBaselineConfig, OpenAIBaselineRunner, build_action_tools
from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.env.environment import OSINTEnvironment
from osint_env.env.openenv_compat import Env
from osint_env.env.reward import compute_answer_reward


README_PATH = Path("README.md")
DOCKERFILE_PATH = Path("Dockerfile")
OPENENV_SPEC_PATH = Path("openenv.yaml")
SHARED_CONFIG_PATH = "datasets/fixed_levels/shared_config_fixed_levels.json"
SEED_FILE_PATH = "datasets/fixed_levels/seed_fixed_levels.json"


@dataclass(slots=True)
class ValidationResult:
    name: str
    passed: bool
    details: dict[str, Any]


def _build_environment() -> OSINTEnvironment:
    shared = load_shared_config(SHARED_CONFIG_PATH)
    env_cfg = clone_environment_config(shared.environment)
    env_cfg.seeding = load_seeding_config(SEED_FILE_PATH)
    env_cfg.llm.provider = "mock"
    return OSINTEnvironment(env_cfg)


def check_hf_space_readiness() -> ValidationResult:
    text = README_PATH.read_text(encoding="utf-8")
    has_sdk = "sdk: docker" in text
    has_port = "app_port: 7860" in text
    has_openenv_tag = "- openenv" in text
    client = TestClient(app)
    health = client.get("/healthz")
    dashboard = client.get("/api/environment")
    spec = client.get("/openenv.yaml")
    passed = all(
        [
            README_PATH.exists(),
            DOCKERFILE_PATH.exists(),
            OPENENV_SPEC_PATH.exists(),
            has_sdk,
            has_port,
            has_openenv_tag,
            health.status_code == 200,
            dashboard.status_code == 200,
            spec.status_code == 200,
        ]
    )
    return ValidationResult(
        name="hf_space_readiness",
        passed=passed,
        details={
            "readme_exists": README_PATH.exists(),
            "dockerfile_exists": DOCKERFILE_PATH.exists(),
            "openenv_spec_exists": OPENENV_SPEC_PATH.exists(),
            "has_sdk_docker": has_sdk,
            "has_app_port": has_port,
            "has_openenv_tag": has_openenv_tag,
            "healthz_status": health.status_code,
            "environment_status": dashboard.status_code,
            "openenv_spec_status": spec.status_code,
        },
    )


def check_openenv_spec_compliance() -> ValidationResult:
    env = _build_environment()
    obs = env.reset()
    client = TestClient(app)
    reset = client.post("/openenv/reset", json={"task_index": 0})
    step = client.post(
        "/openenv/step",
        json={
            "session_id": reset.json()["session_id"] if reset.status_code == 200 else "",
            "action_type": "ANSWER",
            "payload": {"answer": "unknown"},
        },
    )
    state = client.get(f"/openenv/state/{reset.json()['session_id']}") if reset.status_code == 200 else None
    passed = all(
        [
            isinstance(env, Env),
            hasattr(env, "reset"),
            hasattr(env, "step"),
            env.name == "OSINTEnvironment",
            env.state_space == "json-observation",
            env.action_space == ["CALL_TOOL", "ADD_EDGE", "ANSWER"],
            env.episode_max_length == env.config.max_steps,
            isinstance(obs.task, dict),
            "question" in obs.task,
            reset.status_code == 200,
            step.status_code == 200,
            state is not None and state.status_code == 200,
        ]
    )
    return ValidationResult(
        name="openenv_spec_compliance",
        passed=passed,
        details={
            "env_class": type(env).__name__,
            "state_space": env.state_space,
            "action_space": list(env.action_space),
            "episode_max_length": env.episode_max_length,
            "task_keys": sorted(obs.task.keys()),
            "reset_status": reset.status_code,
            "step_status": step.status_code,
            "state_status": 0 if state is None else state.status_code,
        },
    )


class _FakeMessage:
    def __init__(self, answer: str):
        self.content = ""
        self.tool_calls = [
            SimpleNamespace(
                id="fake_tool_call_0",
                function=SimpleNamespace(name="submit_answer", arguments=json.dumps({"answer": answer})),
            )
        ]


class _FakeCompletion:
    def __init__(self, answer: str):
        self.choices = [SimpleNamespace(message=_FakeMessage(answer))]
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self.system_fingerprint = "validation_fp"


class _FakeChatCompletions:
    def create(self, **kwargs: Any) -> _FakeCompletion:
        messages = list(kwargs.get("messages", []))
        initial_observation = {}
        for message in messages:
            if message.get("role") == "user":
                try:
                    initial_observation = json.loads(message.get("content", "{}"))
                except json.JSONDecodeError:
                    initial_observation = {}
                break
        task_id = ((initial_observation.get("task") or {}).get("task_id")) or ""
        env = _build_environment()
        task = next((task for task in env.tasks if task.task_id == task_id), None)
        answer = task.answer if task is not None else "unknown"
        return _FakeCompletion(answer)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def _run_fake_baseline_once(output_dir: Path) -> dict[str, Any]:
    config = OpenAIBaselineConfig(
        api_key="validation",
        episodes=3,
        max_steps=4,
        append_leaderboard=False,
        output_path=str(output_dir / "baseline.json"),
        dashboard_path=str(output_dir / "baseline.html"),
        leaderboard_path=str(output_dir / "leaderboard.json"),
        run_name="validation_baseline",
    )
    runner = OpenAIBaselineRunner.__new__(OpenAIBaselineRunner)
    runner.config = config
    runner.client = _FakeOpenAIClient()
    runner.tools = build_action_tools()
    return runner.run()


def check_baseline_reproducibility() -> ValidationResult:
    with tempfile.TemporaryDirectory() as left_dir_name, tempfile.TemporaryDirectory() as right_dir_name:
        left = _run_fake_baseline_once(Path(left_dir_name))
        right = _run_fake_baseline_once(Path(right_dir_name))

    left_signature = {
        "summary": left["summary"],
        "episodes": [
            {
                "task_id": episode["task_id"],
                "task_answer": episode["task_answer"],
                "agent_answer": episode["agent_answer"],
                "success": episode["success"],
                "steps": episode["steps"],
            }
            for episode in left["episodes"]
        ],
    }
    right_signature = {
        "summary": right["summary"],
        "episodes": [
            {
                "task_id": episode["task_id"],
                "task_answer": episode["task_answer"],
                "agent_answer": episode["agent_answer"],
                "success": episode["success"],
                "steps": episode["steps"],
            }
            for episode in right["episodes"]
        ],
    }
    passed = left_signature == right_signature
    return ValidationResult(
        name="baseline_reproducibility",
        passed=passed,
        details={
            "episodes_checked": len(left_signature["episodes"]),
            "left_signature": left_signature,
            "right_signature": right_signature,
        },
    )


def check_task_and_grader_coverage() -> ValidationResult:
    env = _build_environment()
    tasks = env.tasks
    grader_checks: list[dict[str, Any]] = []
    for task in tasks[:3]:
        correct = compute_answer_reward(
            proposed_answer=task.answer,
            task=task,
            pred_edges=list(task.supporting_edges),
            tool_outputs=[],
            step_count=1,
            model=env.reward_model,
        )
        wrong = compute_answer_reward(
            proposed_answer="unknown",
            task=task,
            pred_edges=[],
            tool_outputs=[],
            step_count=1,
            model=env.reward_model,
        )
        grader_checks.append(
            {
                "task_id": task.task_id,
                "support_edges": len(task.supporting_edges),
                "correct_reward": correct.total,
                "wrong_reward": wrong.total,
                "grader_prefers_correct": correct.total > wrong.total,
            }
        )
    passed = len(tasks) >= 3 and all(row["support_edges"] > 0 and row["grader_prefers_correct"] for row in grader_checks)
    return ValidationResult(
        name="task_and_grader_coverage",
        passed=passed,
        details={
            "task_count": len(tasks),
            "grader_checks": grader_checks,
        },
    )


def run_validation_suite() -> dict[str, Any]:
    results = [
        check_hf_space_readiness(),
        check_openenv_spec_compliance(),
        check_baseline_reproducibility(),
        check_task_and_grader_coverage(),
    ]
    passed = all(result.passed for result in results)
    return {
        "passed": passed,
        "checks": [asdict(result) for result in results],
    }
