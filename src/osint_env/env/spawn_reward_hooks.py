from __future__ import annotations

import math


def critical_steps(main_steps: list[int], parallel_subagent_steps: list[list[int]]) -> int:
    """Compute critical-step latency proxy used in Kimi-style PARL shaping.

    For each stage t, we add:
      Smain(t) + max_i Ssub,i(t)
    where Ssub,i(t) is the i-th sub-agent step count for that stage.
    """
    if len(main_steps) != len(parallel_subagent_steps):
        raise ValueError("main_steps and parallel_subagent_steps must have the same length")

    total = 0
    for stage_main, stage_sub in zip(main_steps, parallel_subagent_steps):
        main = max(0, int(stage_main))
        longest_sub = max((max(0, int(v)) for v in stage_sub), default=0)
        total += main + longest_sub
    return total


def parl_style_spawn_reward(
    task_outcome_reward: float,
    spawn_count: int,
    finished_subtasks: int,
    critical_steps: int,
    lambda_parallel: float = 0.15,
    lambda_finish: float = 0.20,
    anneal: float = 1.0,
    breadth: int | None = None,
    depth: int | None = None,
    max_parallel_hint: int | None = None,
) -> float:
    """Kimi K2.5 inspired PARL reward utility for future multi-agent branches.

    This helper intentionally does not orchestrate agents. It only exposes the reward shape:

      r_parl = r_perf + a * (lambda_parallel * r_parallel + lambda_finish * r_finish + r_latency)

    where:
    - r_parallel encourages non-zero agent spawning (avoids serial collapse)
    - r_finish rewards meaningful completion, preventing spawn-only reward hacking
    - r_latency favors lower critical-step execution paths

    The optional breadth/depth controls are small shaping terms for future branches where
    orchestration state includes tree shape telemetry.
    """
    spawn_count = max(0, int(spawn_count))
    finished_subtasks = max(0, int(finished_subtasks))
    critical_steps = max(1, int(critical_steps))
    anneal = max(0.0, min(1.0, anneal))
    lambda_parallel = max(0.0, float(lambda_parallel))
    lambda_finish = max(0.0, float(lambda_finish))
    breadth = max(0, int(breadth or 0))
    depth = max(0, int(depth or 0))
    max_parallel_hint = max(0, int(max_parallel_hint or 0))

    if spawn_count == 0:
        r_parallel = 0.0
        r_finish = 0.0
    else:
        # Saturating incentive for parallelism so reward cannot grow unbounded with spawns.
        r_parallel = math.tanh(spawn_count / 4.0)
        if max_parallel_hint > 0:
            utilization = min(1.0, spawn_count / max_parallel_hint)
            r_parallel *= (0.7 + (0.3 * utilization))

        r_finish = min(1.0, finished_subtasks / spawn_count)

    if breadth > 0:
        breadth_bonus = 0.04 * math.tanh(breadth / 6.0)
    else:
        breadth_bonus = 0.0

    if depth > 0:
        # Mild depth penalty discourages brittle over-decomposition chains.
        depth_penalty = -0.03 * math.tanh(max(0, depth - 1) / 4.0)
    else:
        depth_penalty = 0.0

    # Optional latency shaping hook using critical steps (higher is worse).
    r_latency = 0.05 * (1.0 / critical_steps)

    auxiliary = (
        (lambda_parallel * r_parallel)
        + (lambda_finish * r_finish)
        + r_latency
        + breadth_bonus
        + depth_penalty
    )
    return float(task_outcome_reward) + (anneal * auxiliary)
