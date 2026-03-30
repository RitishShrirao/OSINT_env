# OSINT RL Environment (MVP)

A professional, scalable prototype of a simulated multi-platform information ecosystem where LLM agents discover, link, and reason over fragmented synthetic data using tools and structured memory.

## Features
- Synthetic dataset generation from hidden canonical graph with aliases/noise
- Three mock platforms: microblog, forum, profile
- Tool surface for search and retrieval across platforms
- OpenEnv-like episode loop with actions: `CALL_TOOL`, `ADD_EDGE`, `ANSWER`
- In-memory knowledge graph and semantic retrieval memory
- Reward shaping (tool efficiency, linking correctness, final answer)
- Single-agent baseline and evaluation metrics

## Quick Start
```bash
source ~/test/bin/activate
uv pip install -e .
osint-env demo
osint-env eval --episodes 20
```

This environment is implemented on top of the Hugging Face `openenv` package (`openenv.env.Env`) and follows the reset/step interaction contract.

## Architecture
```text
src/osint_env/
  domain/         # entities, actions, observations, tasks
  data/           # generator + noisy projections
  platforms/      # mock platform data + tool APIs
  memory/         # KG + semantic index
  env/            # episode state machine + rewards
  agents/         # baseline single-agent orchestrator
  llm/            # pluggable LLM client interfaces
  eval/           # metrics + evaluation runner
  cli.py          # entrypoints
```

## Scalability Notes
- Strong module boundaries to support multi-agent orchestration.
- Configurable generation knobs: users, alias density, noise, red herring rate.
- Deterministic seeds for reproducible benchmark instances.
- LLM provider abstraction for local (Ollama) and hosted backends.
