from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from typing import Any

DEFAULT_HF_JOB_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


def _is_true(value: str | None) -> bool:
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "y", "on"}


def _default_train_output_dir(bucket_name: str | None, run_name: str) -> str:
    if bucket_name:
        return f"/training-outputs/{run_name}"
    return f"artifacts/{run_name}"


def _require_hf_token(value: str | None) -> str:
    token = str(value or "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to launch a Hugging Face Job. "
            "Set HF_TOKEN in your environment or pass --hf-token."
        )
    return token


def _resolve_job_image(job_image: str | None, space_id: str | None) -> str:
    image = str(job_image or "").strip()
    if image:
        return image
    space = str(space_id or "").strip()
    if space:
        return f"hf.co/spaces/{space}"
    return DEFAULT_HF_JOB_IMAGE


def _train_self_play_command(
    *,
    env_config_path: str,
    train_config_path: str,
    output_dir: str,
    dry_run: bool,
) -> list[str]:
    command = [
        "osint-env",
        "train-self-play",
        "--config",
        env_config_path,
        "--train-config",
        train_config_path,
        "--train-output-dir",
        output_dir,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _build_job_command(
    *,
    env_config_path: str,
    train_config_path: str,
    output_dir: str,
    dry_run: bool,
    repo_url: str,
    repo_ref: str,
    repo_subdir: str,
    setup_command: str,
) -> list[str]:
    train_command = _train_self_play_command(
        env_config_path=env_config_path,
        train_config_path=train_config_path,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    repo = str(repo_url).strip()
    if not repo:
        return train_command

    worktree = "/workspace/osint_env_app"
    clone_command = f"git clone --depth 1 {shlex.quote(repo)} {shlex.quote(worktree)}"
    ref = str(repo_ref).strip()
    if ref:
        clone_command = (
            f"git clone --depth 1 --branch {shlex.quote(ref)} "
            f"{shlex.quote(repo)} {shlex.quote(worktree)}"
        )

    shell_lines = [
        "set -euo pipefail",
        "export PYTHONUNBUFFERED=1",
        "export PIP_DISABLE_PIP_VERSION_CHECK=1",
        "command -v git >/dev/null 2>&1 || { echo 'git is required when --repo-url is set' >&2; exit 1; }",
        "mkdir -p /workspace",
        clone_command,
        f"cd {shlex.quote(worktree)}",
    ]
    subdir = str(repo_subdir).strip()
    if subdir:
        shell_lines.append(f"cd {shlex.quote(subdir)}")
    shell_lines.extend(
        [
            "python -m pip install --upgrade pip",
            "python -m pip install -e '.[train]'",
        ]
    )
    setup = str(setup_command).strip()
    if setup:
        shell_lines.append(setup)
    shell_lines.append(_shell_join(train_command))
    return ["bash", "-lc", "\n".join(shell_lines)]


def launch_hf_self_play_job(
    *,
    hf_token: str,
    job_image: str,
    env_config_path: str,
    train_config_path: str,
    flavor: str,
    timeout: str,
    output_dir: str,
    space_id: str = "",
    namespace: str = "",
    run_name: str = "",
    dry_run: bool = False,
    wait: bool = False,
    output_bucket: str = "",
    repo_url: str = "",
    repo_ref: str = "",
    repo_subdir: str = "",
    setup_command: str = "",
) -> dict[str, Any]:
    try:
        from huggingface_hub import Volume, fetch_job_logs, inspect_job, login, run_job
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to launch HF Jobs. "
            "Install dependencies that include huggingface_hub first."
        ) from exc

    token = _require_hf_token(hf_token)
    image = _resolve_job_image(job_image=job_image, space_id=space_id)
    login(token=token, add_to_git_credential=False)

    command = _build_job_command(
        env_config_path=env_config_path,
        train_config_path=train_config_path,
        output_dir=output_dir,
        dry_run=dry_run,
        repo_url=repo_url,
        repo_ref=repo_ref,
        repo_subdir=repo_subdir,
        setup_command=setup_command,
    )

    secrets = {"HF_TOKEN": token}
    for secret_name in ("WANDB_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"):
        secret_value = str(os.getenv(secret_name, "")).strip()
        if secret_value:
            secrets[secret_name] = secret_value

    env: dict[str, str] = {
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if run_name:
        env["OSINT_HF_JOB_RUN_NAME"] = run_name
    for env_name in (
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        "OSINT_TRAIN_STRICT_ASSERTS",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    ):
        env_value = str(os.getenv(env_name, "")).strip()
        if env_value:
            env[env_name] = env_value

    volumes: list[Any] = []
    if output_bucket:
        volumes.append(Volume(type="bucket", source=output_bucket, mount_path="/training-outputs"))

    job = run_job(
        image=image,
        command=command,
        flavor=flavor,
        timeout=timeout,
        namespace=namespace or None,
        env=env,
        secrets=secrets,
        volumes=volumes or None,
    )

    payload: dict[str, Any] = {
        "job_id": str(job.id),
        "job_url": str(job.url),
        "job_image": image,
        "flavor": flavor,
        "timeout": timeout,
        "output_dir": output_dir,
        "output_bucket": output_bucket,
        "repo_url": repo_url,
        "repo_ref": repo_ref,
        "repo_subdir": repo_subdir,
        "space_id_compat": space_id,
        "dry_run": dry_run,
        "waited": False,
    }

    if wait:
        terminal_states = {"COMPLETED", "ERROR", "CANCELLED", "TIMEOUT"}
        last_stage = ""
        while True:
            info = inspect_job(job_id=job.id)
            stage = str(getattr(getattr(info, "status", None), "stage", "") or "")
            if stage != last_stage:
                print(json.dumps({"job_id": str(job.id), "stage": stage, "url": str(job.url)}))
                last_stage = stage
            if stage in terminal_states:
                payload["waited"] = True
                payload["final_stage"] = stage
                if stage != "COMPLETED":
                    payload["logs"] = list(fetch_job_logs(job_id=job.id))
                break
            time.sleep(15)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch OSINT self-play training as a separate Hugging Face Job on dedicated compute."
    )
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="HF token. Defaults to HF_TOKEN env var.")
    parser.add_argument(
        "--job-image",
        default=os.getenv("HF_JOB_IMAGE", ""),
        help=(
            "Docker image for the dedicated training job. "
            f"Defaults to {DEFAULT_HF_JOB_IMAGE!r} unless --space-id is provided."
        ),
    )
    parser.add_argument(
        "--space-id",
        default=os.getenv("HF_SPACE_ID", ""),
        help="Optional compatibility fallback to reuse a Space image, e.g. owner/space-name.",
    )
    parser.add_argument(
        "--env-config",
        default=os.getenv("TRAIN_ENV_CONFIG_PATH", "config/shared_config.json"),
        help="Environment config path inside the training image or checked-out repo.",
    )
    parser.add_argument(
        "--train-config",
        default=os.getenv("TRAIN_SELF_PLAY_CONFIG_PATH", "config/self_play_training_hf_l40s_full.json"),
        help="Training config path inside the training image or checked-out repo.",
    )
    parser.add_argument("--flavor", default=os.getenv("HF_JOB_FLAVOR", "l40s"))
    parser.add_argument("--timeout", default=os.getenv("HF_JOB_TIMEOUT", "8h"))
    parser.add_argument("--namespace", default=os.getenv("HF_JOB_NAMESPACE", ""))
    parser.add_argument("--run-name", default=os.getenv("HF_JOB_RUN_NAME", "osint-self-play-job"))
    parser.add_argument("--output-bucket", default=os.getenv("HF_JOB_OUTPUT_BUCKET", ""))
    parser.add_argument("--output-dir", default=os.getenv("TRAIN_SELF_PLAY_OUTPUT_DIR", ""))
    parser.add_argument(
        "--repo-url",
        default=os.getenv("HF_JOB_REPO_URL", ""),
        help="Optional git repository URL to clone inside the job before training.",
    )
    parser.add_argument(
        "--repo-ref",
        default=os.getenv("HF_JOB_REPO_REF", ""),
        help="Optional git branch, tag, or commit-ish to check out when --repo-url is used.",
    )
    parser.add_argument(
        "--repo-subdir",
        default=os.getenv("HF_JOB_REPO_SUBDIR", ""),
        help="Optional subdirectory inside the cloned repo that contains pyproject.toml.",
    )
    parser.add_argument(
        "--setup-command",
        default=os.getenv("HF_JOB_SETUP_COMMAND", ""),
        help="Optional shell command to run after install and before training.",
    )
    parser.add_argument("--dry-run", action="store_true", default=_is_true(os.getenv("RUN_SELF_PLAY_DRY_RUN", "")))
    parser.add_argument("--wait", action="store_true", default=_is_true(os.getenv("HF_JOB_WAIT", "")))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_name = str(args.run_name).strip() or "osint-self-play-job"
    output_bucket = str(args.output_bucket).strip()
    output_dir = str(args.output_dir).strip() or _default_train_output_dir(output_bucket, run_name)

    payload = launch_hf_self_play_job(
        hf_token=str(args.hf_token),
        job_image=str(args.job_image),
        env_config_path=str(args.env_config),
        train_config_path=str(args.train_config),
        flavor=str(args.flavor),
        timeout=str(args.timeout),
        output_dir=output_dir,
        space_id=str(args.space_id),
        namespace=str(args.namespace),
        run_name=run_name,
        dry_run=bool(args.dry_run),
        wait=bool(args.wait),
        output_bucket=output_bucket,
        repo_url=str(args.repo_url),
        repo_ref=str(args.repo_ref),
        repo_subdir=str(args.repo_subdir),
        setup_command=str(args.setup_command),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
