---
title: OSINT OpenEnv
emoji: 🕵️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
tags:
  - openenv
  - osint
  - benchmark
  - docker
  - fastapi
short_description: Docker OSINT benchmark with fixed OpenEnv tasks.
---

# OSINT OpenEnv

OSINT OpenEnv is a synthetic benchmark environment for tool-using agents that must recover identities, trace events, and link entities across noisy multi-platform records. The project is designed to feel like a compact OSINT workflow rather than a raw QA dataset: agents query mock profiles, posts, forum threads, and semantic memory, build a working graph, and then submit an answer.

The motivation is to provide a reproducible OpenEnv-compatible environment for evaluating graph-building and tool-using reasoning without depending on live web data, unstable APIs, or private corpora. That makes it useful for local development, regression testing, and hosted demos such as a Docker-based Hugging Face Space.

## Environment Summary

The environment generates or loads a hidden canonical graph of users, aliases, organizations, locations, posts, threads, and events. It then exposes partial platform views and a task list drawn from that graph.

The default hosted Space uses the fixed-level benchmark in `datasets/fixed_levels/seed_fixed_levels.json`, which now contains 30 stable tasks over a larger shared seeded graph.

The repository now supports two dataset backends:

- `canonical` (existing fixed-level synthetic graph pipeline)
- `metaqa` (MetaQA KB + QA files for `1-hop`, `2-hop`, and `3-hop`)

Use `config/shared_config.json` or CLI flags (`--dataset-mode`, `--metaqa-root`, `--metaqa-hops`, `--metaqa-splits`) to choose which backend to run.

## Action Space

The environment exposes three actions:

- `CALL_TOOL`: query platform views or semantic memory such as `search_posts`, `get_profile`, `search_threads`, `get_connections`, or `search_memory`.
- `ADD_EDGE`: add a candidate relation to the working memory graph.
- `ANSWER`: submit the final answer as an exact node id string.

## Observation Space

Each step returns a JSON observation with four parts:

- `tool_outputs`: the most recent tool results.
- `graph_snapshot`: the current working-memory graph edges.
- `action_history`: recent actions and rewards.
- `task`: the active task id, task type, and question.

## Task Types And Difficulty

The benchmark mixes direct lookups with multi-hop traces:

- Easy: single-hop identity resolution, organization lookup, event lookup, or location lookup.
- Mid: two-hop alias-to-user-to-organization or thread-to-event-to-user traces.
- High: cross-platform multi-hop traces combining aliases, authored content, event references, organization links, and direct connections.

In MetaQA mode, hop buckets are mapped into the same reward difficulty tiers:

- `1-hop` -> `easy`
- `2-hop` -> `medium`
- `3-hop` -> `hard`

Common task families include:

- `identity_resolution`
- `network_discovery`
- `event_tracing`
- `cross_platform_linking`
- `deanonymization`
- `convoluted_trace`

Expected difficulty increases with the number of relations the agent must chain together and whether the evidence is split across posts, threads, aliases, and profile edges.

## Repository Layout

```text
src/osint_env/
  agents/        single-agent and swarm runners
  baselines/     reusable OpenAI baseline runner
  config/        shared config and seed loading
  data/          graph/view/task generation
  domain/        dataclasses and environment models
  env/           environment, reward logic, OpenEnv compatibility shim
  eval/          evaluation metrics and leaderboard helpers
  llm/           mock, Ollama, and OpenAI client wrappers
  memory/        working graph and semantic memory
  platforms/     tool APIs over synthetic platform views
  viz/           dashboard export

scripts/
  build_fixed_levels_dataset.py
  run_openai_baseline.py

datasets/fixed_levels/
  seed_fixed_levels.json
  shared_config_fixed_levels.json
  qwen_swarm_benchmark_fixed_levels.json

server.py        FastAPI app for local use and Docker/HF Spaces
Dockerfile       Container entrypoint for Hugging Face Docker Spaces
```

## Setup

Python 3.10+ is required.

Local install:

```bash
python -m pip install -e .
```

Install optional adversarial self-play training stack (TRL + Transformers):

```bash
python -m pip install -e ".[train]"
```

Run tests:

```bash
python -m pytest -q
```

Run the automated release gate:

```bash
python scripts/validate_release.py
```

## Usage

Run one demo episode:

```bash
osint-env demo --agent-mode swarm --llm-provider mock
```

Run against MetaQA using the provided sample config:

```bash
osint-env demo --config config/shared_config_metaqa.json --dataset-mode metaqa --llm-provider mock
```

Run MetaQA with only selected hop buckets:

```bash
osint-env eval --config config/shared_config_metaqa.json --dataset-mode metaqa --metaqa-hops 1-hop,2-hop --episodes 5 --llm-provider mock
```

Run a quick evaluation:

```bash
osint-env eval --episodes 5 --agent-mode swarm --llm-provider mock
```

Export a dashboard:

```bash
osint-env benchmark --episodes 5 --agent-mode swarm --llm-provider mock --name quick_check
```

Run Kimi-style adversarial self-play scaffold (dry-run by default in the example config):

```bash
osint-env train-self-play --config config/shared_config.json --train-config config/self_play_training_example.json --dry-run
```

When you have compute and the train dependencies installed, remove `--dry-run` (or set `"dry_run": false` in the training config) to execute TRL GRPO updates for alternating generator and answerer phases.

The training config also supports `"model_topology": "dual"|"shared"`, `"phase_schedule": "generator_answerer"|"answerer_generator_answerer"`, `"tuning_mode": "full"|"lora"`, and `"canonical_graph_mode": "generate"|"fixed"` so you can switch between two-model vs single-model self-play, full fine-tuning vs LoRA adapters, and whether canonical graph structure is generated each round or kept fixed while training question/answer behavior.

### Hugging Face Space Smoke Run (Qwen 3.5 0.8B + W&B)

For a short verification run (enough to confirm W&B logging before scaling up), use:

```bash
osint-env train-self-play --config config/shared_config.json --train-config config/self_play_training_hf_a10g_smoke.json
```

This config:

- uses `Qwen/Qwen3.5-0.8B`
- enables W&B reporting (`wandb_enabled: true`)
- uses `pipeline_mode: "swarm_v2"` with `canonical_graph_mode: "fixed"` to keep canonical graph candidates stable while training question/answer behavior
- keeps training intentionally short (`rounds=1`, `max_steps=5` per phase)
- uses LoRA with small batch settings so it can run as a smoke test on an A10G

To enable canonical graph generation during swarm_v2 training, switch `"canonical_graph_mode"` to `"generate"` in the training config.

Space setup checklist:

1. In Space **Settings -> Hardware**, select **NVIDIA A10G (large)**.
2. In Space **Settings -> Variables and secrets**, set `WANDB_API_KEY`.
3. Optionally set `WANDB_ENTITY` if your project belongs to a team.
4. Install training extras in the Space environment: `python -m pip install -e ".[train]"`.

W&B run naming is controlled by `wandb_run_name_prefix` and will emit phase-specific runs like `...-r001-generator` and `...-r001-answerer`.

### Reward Functions In Self-Play (Generator + Answerer)

Self-play trains two policies with role-specific reward functions defined in `src/osint_env/training/rewards.py`.

Generator reward (`GeneratorRewardFunction`) and answerer reward (`AnswererRewardFunction`) are both returned to GRPO as scalar scores per completion, and both are clipped to a stable range before optimization.

#### Generator Reward (Task-Proposing Agent)

The generator is rewarded for producing valid, grounded, diverse, and hard tasks.

In `legacy` pipeline mode, the reward is a weighted sum:

- `validity`: checks non-empty `question`, non-empty `answer`, and bounded `supporting_edges`.
- `hardness`: uses a frozen answerer judge; reward is higher when the judge gets the generated question wrong.
- `diversity`: penalizes near-duplicate questions via token-level Jaccard similarity against prior generated questions.
- `consistency`: checks that support edges exist in the canonical graph and that the answer/question are graph-grounded.

Default weights (configurable through `generator_reward_weights` in training config):

- `validity`: `0.35`
- `hardness`: `0.45`
- `diversity`: `0.10`
- `consistency`: `0.10`

In `swarm_v2` pipeline mode, generation uses strict replay/validation first, then a structured reward:

- Hard-gated validation via `SwarmV2ReplayValidator` (invalid samples get a fixed negative reward path).
- Reward components include validity, derivability/replayability, hardness, swarm diversity, shared-context pressure targeting, and PARL-inspired orchestration bonuses (`parallel` + `finish`).
- Invalid or non-replayable candidates are penalized before the weighted positive terms are applied.

#### Answerer Reward (Question-Solving Agent)

The answerer reward wraps environment-native grading so train-time behavior matches benchmark-time incentives.

For each completion, `AnswererRewardFunction`:

- extracts the predicted answer from completion text/JSON,
- reconstructs a transient `TaskInstance` from row fields (`question`, `answer`, `supporting_edges_json`, `difficulty`),
- optionally extracts predicted supporting edges from JSON or text,
- calls `compute_answer_reward(...)` from `src/osint_env/env/reward.py`.

`compute_answer_reward` combines exact answer quality with graph-utility shaping:

- output format validity and exact correctness,
- knowledge-carrier and knowledge-indexing utility,
- connectivity and supporting-edge F1 against task support edges,
- efficiency and compactness penalties,
- relation/entity informativeness and repetition control (difficulty-dependent).

Difficulty controls (`easy`, `medium`, `hard`) are preserved during training exactly as in the environment scorer, so the answerer sees the same tiered reward profile used in evaluation.

In `swarm_v2`, the answerer reward also adds PARL-style orchestration credit (spawn/finish behavior) on top of base answer reward when orchestrator telemetry is present in the completion payload.

Detailed design notes are in `docs/adversarial_self_play.md`.

## OpenAI Baseline

The reproducible OpenAI baseline is implemented in `scripts/run_openai_baseline.py`. It runs on the fixed-level benchmark, uses a stable seeded graph/task set, writes a JSON artifact, appends a leaderboard record, and exports a dashboard.

Default behavior:

- dataset: fixed-level benchmark
- episodes: 30
- max steps per episode: 8
- temperature: 0.0
- output artifact: `artifacts/baselines/openai_fixed_levels_latest.json`

Run it with an API key:

```bash
export OPENAI_API_KEY="your_key_here"
python scripts/run_openai_baseline.py --model gpt-5-nano
```

The script is designed to stay bounded enough for a normal benchmark pass to finish comfortably under 20 minutes on a lightweight chat model, while still using the full fixed task set. For repeatability it fixes the benchmark graph/tasks and uses deterministic decoding settings. Because remote model backends can still change over time, the output artifact also records model metadata and system fingerprints when available.

## Inference Script

The submission-ready inference entrypoint is the root `inference.py` file. It talks to the deployed Hugging Face Space over HTTP, uses the OpenAI client for all model calls, and emits structured stdout logs in the `[START]`, `[STEP]`, and `[END]` format.

The script accepts `HF_TOKEN` as the primary auth variable and also supports `OPENAI_API_KEY` or `API_KEY` as local fallbacks.
After a successful run, `inference.py` also posts the evaluation summary back to the Space so the latest `/dashboard` view reflects that run.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional environment variables:

- `SPACE_URL` default: `https://siddeshwar1625-osint.hf.space`
- `TASK_INDICES` default: `0,10,20`
- `MAX_STEPS` default: `8`

Example local test command against a running local server:

```bash
API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-5.4-mini OPENAI_API_KEY=your_key SPACE_URL=http://127.0.0.1:7860 python inference.py
```

Example test command against the deployed Space:

```bash
API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-5.4-mini OPENAI_API_KEY=your_key SPACE_URL=https://siddeshwar1625-osint.hf.space python inference.py
```

## Docker And Hugging Face Space

The repository is ready for a Docker-based Hugging Face Space:

- `README.md` includes `sdk: docker`
- `README.md` includes the `openenv` Space tag
- `Dockerfile` serves `server.py` on port `7860`

Local Docker smoke test:

```bash
docker build -t osint-openenv .
docker run --rm -p 7860:7860 osint-openenv
```

Then open `http://localhost:7860`.

The FastAPI app serves:

- `/`: overview page
- `/dashboard`: generated benchmark dashboard
- `/api/environment`: environment metadata
- `/health`: health check (validator-friendly alias)
- `/healthz`: health check (legacy alias)
- `/openenv.yaml`: OpenEnv HTTP spec stub
- `/openenv/tasks`: task enumeration
- `/reset` and `/openenv/reset`: episode reset endpoints
- `/step` and `/openenv/step`: episode step endpoints
- `/state` and `/openenv/state/{session_id}`: session state endpoints (`/state` returns the latest session)

## Automated Validation

The repository includes a pass/fail validation gate for the core delivery requirements:

- Hugging Face Space readiness
- OpenEnv spec compliance
- reproducible baseline behavior
- at least 3 fixed tasks with working graders
- Docker image build in CI

Local gate:

```bash
python scripts/validate_release.py
```

CI gate:

- `.github/workflows/validation.yml`
- runs `pytest`
- runs the validation script
- runs `docker build`

## Baseline Scores

The fixed-level benchmark was expanded from the earlier 15-question set to a 30-question set with a larger seeded graph, so older benchmark artifacts should be treated as legacy and regenerated on the new dataset before using them as reference scores.

After you supply an OpenAI API key, the current baseline scores for the expanded benchmark will be written to:

- `artifacts/baselines/openai_fixed_levels_latest.json`
- `artifacts/baselines/openai_fixed_levels_dashboard.html`

## Notes On `pyproject.toml`

The packaging file is structurally correct for a `src/` layout and editable installs. The main gaps were deployment/runtime related rather than build-breaking:

- `openenv` is now version-bounded explicitly.
- `fastapi` and `uvicorn` are included because the repo now ships a real web server.
- pytest is pointed at the `tests/` directory, and the test suite also adds `src/` to `sys.path` so source-layout imports work reliably during local runs.

## Development Notes

The project keeps a lightweight local compatibility shim for `openenv` so the source tree remains importable even before dependencies are installed. In a normal install or Docker build, the real `openenv` package from PyPI is still used.
