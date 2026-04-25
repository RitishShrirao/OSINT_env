# Adversarial Self-Play Training (Kimi-Style + TRL)

This repository now includes a code scaffold for alternating adversarial self-play with Hugging Face TRL.

## Goal

Train two policies in alternating rounds:

- Generator policy: proposes hard OSINT tasks (question + answer + supporting edges).
- Answerer policy: solves tasks proposed by the generator.

The loop is intended to move from static evaluation toward on-policy co-evolution.

## Kimi-style Objective Mapping

The implementation maps the requested Kimi-style ingredients onto TRL GRPO as follows:

- Grouped rollouts: `num_generations` in each GRPO phase.
- Relative reward baseline: GRPO group-relative advantages.
- Clipped policy updates: `epsilon` clipping in GRPO objective.
- KL/reference regularization: `beta` in GRPOConfig.
- Token-level online RL behavior: GRPO online generation with reward functions.
- Toggle schedule: explicit alternating generator and answerer rounds.

## Topology and Scheduling Options

- `model_topology: "dual"`: train separate generator and answerer models.
- `model_topology: "shared"`: train one shared model for both roles.
	- Use `shared_model_name_or_path` to set the common base checkpoint.
- `phase_schedule: "generator_answerer"`: default two-phase loop per round.
- `phase_schedule: "answerer_generator_answerer"`: solver-first curriculum:
	1. Train answerer on current adversarial pool.
	2. Freeze that answerer snapshot while training generator against it.
	3. Train answerer again on newly generated adversarial tasks.

This directly supports the "train solver, freeze, attack, retrain solver" sequence.

## Canonical Graph Mode

- `canonical_graph_mode: "generate"` (default): generator can propose canonical graph updates in `swarm_v2`.
- `canonical_graph_mode: "fixed"`: canonical graph candidates are held fixed per prompt, so training focuses on question/answer behavior over stable graph structure.

## Tuning Modes

- `tuning_mode: "full"`: full-model GRPO fine-tuning.
- `tuning_mode: "lora"`: PEFT LoRA adapters for GRPO updates.
	- Configure via `lora` block: `r`, `alpha`, `dropout`, `target_modules`, `bias`, `task_type`.

## Reward Design

### Generator (adversarial swarm)

`GeneratorRewardFunction` combines weighted components:

- Validity: checks parsable task fields and bounded support-edge size.
- Hardness: rewards questions the frozen answerer currently gets wrong.
- Diversity: penalizes near-duplicate questions via token-overlap similarity.
- Consistency: rewards edge/answer/question grounding against canonical graph context.

Weights are configurable in `generator_reward_weights`.

### Answerer (existing reward integration)

`AnswererRewardFunction` wraps existing environment reward logic:

- Reuses `compute_answer_reward` from `src/osint_env/env/reward.py`.
- Builds transient `TaskInstance` objects from training rows.
- Preserves difficulty-aware reward behavior (`easy` / `medium` / `hard`).

## Entry Points

- CLI command: `osint-env train-self-play`
- Main runner: `src/osint_env/training/self_play.py`
- Config loader: `src/osint_env/training/config.py`
- Reward functions: `src/osint_env/training/rewards.py`
- Example config: `config/self_play_training_example.json`

## Dry Run Mode

The example config sets `dry_run: true` by default.

In dry run mode, the pipeline still:

- Materializes generator/answerer datasets per round.
- Materializes optional `answerer_pre_dataset` when using solver-first schedule.
- Produces generated-task artifacts (fallback generator path).
- Writes a full run summary.

But it skips expensive GRPO updates.

## Compute Mode

When compute is available:

1. Install train dependencies: `python -m pip install -e ".[train]"`
2. Disable dry run (`--dry-run` off and/or `"dry_run": false` in config).
3. Run `osint-env train-self-play`.

Outputs are written under `artifacts/self_play` unless overridden.
