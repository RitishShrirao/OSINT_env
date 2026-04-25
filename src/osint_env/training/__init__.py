"""Adversarial self-play training helpers."""

from osint_env.training.config import (
    GeneratorRewardWeights,
    KimiGRPOPhaseConfig,
    LoraTuningConfig,
    SelfPlayTrainingConfig,
    SwarmV2Config,
    SwarmV2SharedContextConfig,
    SwarmV2SwarmConfig,
    SwarmV2ValidationConfig,
    load_self_play_config,
)
from osint_env.training.hf_jobs import launch_hf_self_play_job
from osint_env.training.self_play import run_adversarial_self_play

__all__ = [
    "GeneratorRewardWeights",
    "KimiGRPOPhaseConfig",
    "LoraTuningConfig",
    "SelfPlayTrainingConfig",
    "SwarmV2Config",
    "SwarmV2SharedContextConfig",
    "SwarmV2SwarmConfig",
    "SwarmV2ValidationConfig",
    "load_self_play_config",
    "launch_hf_self_play_job",
    "run_adversarial_self_play",
]
