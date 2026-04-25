from osint_env.training.hf_jobs import (
    DEFAULT_HF_JOB_IMAGE,
    _build_job_command,
    _default_train_output_dir,
    _resolve_job_image,
)


def test_resolve_job_image_prefers_explicit_image():
    assert _resolve_job_image("python:3.12", "owner/space") == "python:3.12"


def test_resolve_job_image_supports_space_fallback():
    assert _resolve_job_image("", "owner/space") == "hf.co/spaces/owner/space"
    assert _resolve_job_image("", "") == DEFAULT_HF_JOB_IMAGE


def test_default_train_output_dir_uses_bucket_mount_when_present():
    assert _default_train_output_dir("my-bucket", "run-42") == "/training-outputs/run-42"
    assert _default_train_output_dir("", "run-42") == "artifacts/run-42"


def test_build_job_command_runs_train_directly_when_image_has_code():
    command = _build_job_command(
        env_config_path="config/shared_config.json",
        train_config_path="config/train.json",
        output_dir="artifacts/self_play",
        dry_run=False,
        repo_url="",
        repo_ref="",
        repo_subdir="",
        setup_command="",
    )
    assert command == [
        "osint-env",
        "train-self-play",
        "--config",
        "config/shared_config.json",
        "--train-config",
        "config/train.json",
        "--train-output-dir",
        "artifacts/self_play",
    ]


def test_build_job_command_bootstraps_repo_when_requested():
    command = _build_job_command(
        env_config_path="config/shared_config.json",
        train_config_path="config/train.json",
        output_dir="/training-outputs/run-1",
        dry_run=True,
        repo_url="https://github.com/example/osint-env.git",
        repo_ref="main",
        repo_subdir=".",
        setup_command="python -m pip install flash-attn --no-build-isolation",
    )
    assert command[:2] == ["bash", "-lc"]
    script = command[2]
    assert "git clone --depth 1 --branch main https://github.com/example/osint-env.git /workspace/osint_env_app" in script
    assert "python -m pip install -e '.[train]'" in script
    assert "python -m pip install flash-attn --no-build-isolation" in script
    assert "--train-config config/train.json" in script
    assert "--train-output-dir /training-outputs/run-1" in script
    assert "--dry-run" in script
