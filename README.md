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

Run a quick evaluation:

```bash
osint-env eval --episodes 5 --agent-mode swarm --llm-provider mock
```

Export a dashboard:

```bash
osint-env benchmark --episodes 5 --agent-mode swarm --llm-provider mock --name quick_check
```

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
- `/healthz`: health check
- `/openenv.yaml`: OpenEnv HTTP spec stub
- `/openenv/tasks`: task enumeration
- `/openenv/reset`: episode reset endpoint
- `/openenv/step`: episode step endpoint
- `/openenv/state/{session_id}`: current session state endpoint

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
