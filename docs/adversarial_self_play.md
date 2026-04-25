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

For `swarm_v2`, the reward now prioritizes:

- Valid, replayable task structure first.
- Hardness against the frozen answerer second.
- Diversity and compact multi-agent/shared-context usage after validity.

This avoids the degenerate regime where almost every sample is invalid and the whole batch stays negative.

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

## Post-Training Evaluation

After a non-dry-run training job completes, the runner now writes a post-training evaluation artifact that:

- Uses the finetuned generator to create fresh evaluation questions.
- Evaluates both the finetuned answerer and the original/base answerer on those generated questions.
- Reports a `delta_vs_original` summary so you can see whether fine-tuning actually improved task success, reward, and graph F1.
- Saves the summary and episode rows under `post_training_evaluation.json`.

You can control this flow with:

- `generated_task_max_new_tokens`: decoding budget for generator-side task sampling/eval.
- `post_training_eval_questions`: how many fresh tasks to evaluate after training.
- `post_training_eval_answer_max_new_tokens`: answerer decoding budget for the final eval pass.

## Checkpoints And Final Models

Self-play outputs are written under `output_dir` (default `artifacts/self_play`) unless overridden.

Per round and phase you will now find:

- `round_XXX/<phase>/checkpoint-*`: intermediate trainer checkpoints saved every `save_steps`.
- `round_XXX/<phase>/final_model`: final saved model for that phase, with tokenizer files.
- `self_play_summary.json`: top-level run summary.
- `post_training_evaluation.json`: generated-question evaluation written after training.

## Compute Mode

When compute is available:

1. Install train dependencies: `python -m pip install -e ".[train]"`
2. Disable dry run (`--dry-run` off and/or `"dry_run": false` in config).
3. Run `osint-env train-self-play`, or launch a dedicated Hugging Face Job with `osint-env-launch-hf-job` if you want the Space to stay on CPU while training runs on separate GPU compute.

Outputs are written under `artifacts/self_play` unless overridden.

## Standalone Server Script

For an SSH server or other standalone machine, you can use `scripts/train_self_play_standalone.sh`.

Example:

```bash
VENV_PATH="$HOME/arl" \
INSTALL_TRAIN_DEPS=1 \
TRAIN_ENV_CONFIG_PATH="config/shared_config.json" \
TRAIN_SELF_PLAY_CONFIG_PATH="config/self_play_training_hf_a10g_smoke.json" \
TRAIN_SELF_PLAY_OUTPUT_DIR="artifacts/self_play_server" \
bash scripts/train_self_play_standalone.sh
```

Useful environment variables:

- `BOOTSTRAP_VENV=1`: create the virtualenv automatically if it does not exist yet.
- `TRAIN_SELF_PLAY_ROUNDS=2`: override the round count without editing JSON.
- `RUN_SELF_PLAY_DRY_RUN=1`: skip GRPO updates and only materialize artifacts.
- `TRAIN_SETUP_COMMAND='python -m pip install flash-attn --no-build-isolation'`: run any host-specific setup before training.
