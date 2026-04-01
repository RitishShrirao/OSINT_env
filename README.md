# OSINT RL Environment

This repository implements a simulated OSINT-style reinforcement learning environment where agents build and query a knowledge graph over fragmented multi-platform synthetic data.

The codebase now supports both single-agent and low-width multi-agent swarm execution, seeded task and graph bootstrapping, benchmark scoring, and interactive visualization.

## 1. What The Project Does

The environment models a realistic workflow for information discovery and linking:

1. Generate a hidden canonical graph with users, aliases, organizations, locations, and links.
2. Project noisy partial views into mock platforms (microblog, forum, profile).
3. Ask identity-resolution, network-discovery, and event-tracing questions.
4. Let agents call tools, add graph edges, and submit answers.
5. Score episodes using a composite reward that combines correctness, retrieval utility, graph quality, and efficiency.

The tool layer also supports semantic-memory retrieval over prior observations:

- search_memory(query, k): vector-style retrieval over accumulated tool outputs.

## 2. Current Capabilities

- Single-agent baseline runner.
- Multi-agent swarm runner with constrained breadth and width (configurable, low by default).
- Seeded graph nodes and edges from user-provided JSON.
- Seeded questions from user-provided JSON.
- LLM-assisted generation hooks for remaining graph/task expansion with deterministic fallback.
- Persistent benchmark leaderboard with composite utility score.
- Interactive dashboard showing:
  - canonical graph,
  - episode graph diff (predicted vs truth),
  - source database explorer,
  - benchmark charts and leaderboard.

## 3. Installation

Environment setup from the project root:

1. Activate your Python environment.
2. Install package dependencies.

Example:

   source ~/arl/bin/activate
   uv pip install -e .

The project requires Python 3.10+.

## 3.1 LLM Backends

The environment supports three LLM providers:

- mock: deterministic fallback for reproducible local tests.
- ollama: local model inference (recommended for offline development).
- openai: remote API provider using an API key.

The provider is configured through config/shared_config.json (llm block) and can be overridden from CLI.

### Local Ollama Setup (Qwen 3 2B)

1. Install Ollama.
2. Start Ollama service.
3. Pull the model:

  ollama pull qwen3:2b

If your local Ollama registry does not expose `qwen3:2b`, use:

  ollama pull qwen3:1.7b
  ollama cp qwen3:1.7b qwen3:2b

4. Run demo in swarm mode with local model:

  osint-env demo --agent-mode swarm --llm-provider ollama --llm-model qwen3:2b

### OpenAI Setup

1. Export API key:

  export OPENAI_API_KEY="your_key_here"

2. Run with OpenAI backend:

  osint-env eval --episodes 10 --llm-provider openai --llm-model gpt-4o-mini

You can also provide the key via config/shared_config.json using llm.openai_api_key,
or specify a custom environment variable name via llm.openai_api_key_env.

## 4. Repository Layout

   src/osint_env/
    agents/        single-agent and swarm runners
    config/        shared config loader
    data/          canonical graph, views, and task generation
    domain/        data models and configuration dataclasses
    env/           OpenEnv environment and reward logic
    eval/          metrics, runner, leaderboard
    llm/           LLM client interface and local mock
    memory/        in-memory KG and semantic memory
    platforms/     platform tool APIs
    viz/           dashboard export
    cli.py         command-line entrypoint

   config/
    shared_config.json   shared runtime/environment/swarm/reward config
    seed_example.json    example seeded graph and question file

## 5. Shared Configuration

All core knobs are centralized in config/shared_config.json.

This file includes:

- environment generation controls,
- swarm limits,
- spawn reward shaping hyperparameters,
- seeding defaults,
- llm backend defaults,
- runtime output paths.

Default swarm settings are intentionally conservative:

- max_agents: 3
- max_breadth: 2
- max_width: 2
- max_depth: 2

These defaults keep orchestration cost and branching low while enabling swarm behavior.

## 6. Seeding Questions And Partial Graphs

You can manually seed:

- graph nodes,
- graph edges,
- task questions (optionally with answers and supporting edges).

Use a seed file with the same structure as config/seed_example.json and pass it using --seed-file.

Workflow:

1. Add your manual graph fragments and questions to a JSON file.
2. Keep llm_generate_remaining_graph and llm_generate_remaining_tasks enabled to fill the rest automatically.
3. Run demo/eval/benchmark with --seed-file.

## 7. CLI Usage

All commands accept:

- --config for shared config path (default: config/shared_config.json)
- --seed-file for seeded graph/task input JSON
- --agent-mode with values: config, single, swarm
- --llm-provider with values: config, mock, ollama, openai
- --llm-model to override configured model
- --ollama-base-url to override local Ollama endpoint
- --openai-api-key or --openai-api-key-env for OpenAI authentication

Main commands:

1. Run one episode:

     osint-env demo --agent-mode swarm

2. Evaluate episodes:

     osint-env eval --episodes 20 --agent-mode single

3. Benchmark and export dashboard:

     osint-env benchmark --episodes 20 --name baseline_swarm

4. Multi-seed benchmark sweep:

     osint-env benchmark-sweep --seeds 7,11,17,23,31 --name-prefix sweep_swarm

5. Print leaderboard:

     osint-env leaderboard --sort-by leaderboard_score --top 15

6. Export explorer without full benchmark:

     osint-env viz --with-demo --output artifacts/osint_explorer.html

  7. Benchmark with local Qwen model:

    osint-env benchmark --episodes 20 --agent-mode swarm --llm-provider ollama --llm-model qwen3:2b --name qwen3_swarm

8. Fast local smoke benchmark:

    osint-env benchmark --episodes 1 --agent-mode swarm --llm-provider ollama --llm-model qwen3:2b --seed-file config/seed_ollama_smoke.json --name ollama_qwen_smoke

## 8. Multi-Agent Swarm Design

Swarm orchestration is implemented in src/osint_env/agents/swarm_agent.py.

Design choices:

- Shared environment state (single episode state machine).
- Planner rounds bounded by max_depth and planner_rounds.
- Parallel workers bounded by min(max_agents, max_breadth, max_width).
- Each worker performs limited tool calls, then attempts edge addition.
- Final answer is submitted once planning rounds complete or episode ends.

Reward compatibility:

- Existing edge and answer reward components are unchanged.
- Spawn utility is added as an auxiliary term using the PARL-style helper in src/osint_env/env/spawn_reward_hooks.py.
- Spawn telemetry (count, critical steps, completion) is tracked in episode info and evaluation summaries.

## 9. Reward Design (Integrated Notes)

The reward function is a composite of graph-construction and answer-time utility terms. It combines ideas from DeepPath, EMNLP 2018 reward shaping, UniRel, and AutoGraph-R1.

### 9.1 Edge Reward During Graph Construction

For each ADD_EDGE action, the environment combines:

1. Global accuracy signal (DeepPath-style positive/negative credit).
2. Soft shaping term inspired by EMNLP 2018 reward shaping:

  R = Rb + (1 - Rb) f(s, r, o)

  where f is approximated in code with relation and type priors plus small domain priors.

3. Efficiency bonus inversely proportional to step count.
4. Diversity bonus using signature novelty against previous edges.
5. Relation informativeness using normalized relation IDF.
6. Entity informativeness using inverse hubness penalty.
7. Connectivity gain bonus for bridge-style edges.

### 9.2 Final Answer Reward

For ANSWER, reward includes:

1. format validity,
2. correctness,
3. knowledge-carrying utility (AutoGraph-style deducibility),
4. knowledge-indexing utility (AutoGraph-style evidence coverage proxy over tool outputs),
5. UniRel-style connectivity score over seed entities,
6. graph F1 against supporting edges,
7. compactness and repetition controls,
8. efficiency and informativeness terms.

### 9.3 Swarm Auxiliary Reward

The swarm runner adds a PARL-style auxiliary term based on:

- spawn parallelism,
- finished subtask ratio,
- critical-step latency proxy,
- optional breadth and depth shaping.

This auxiliary term is configurable in shared_config.json via spawn_reward.

### 9.4 Benchmark Metrics

Evaluation tracks:

- task success,
- graph F1,
- deanonymization accuracy,
- tool efficiency,
- retrieval and structural utility signals,
- spawn signals (for swarm runs),
- composite leaderboard score.

## 10. Interactive Dashboard

Dashboard export includes:

- canonical graph explorer,
- episode graph comparison,
- node and edge inspectors,
- source database table with record detail pane,
- reward and graph traces,
- sortable leaderboard snapshot.

Primary outputs:

- artifacts/osint_dashboard.html
- artifacts/osint_explorer.html
- artifacts/sweep_dashboards/*.html

## 11. Notes On LLM Generation

Dataset generation supports an LLM-assisted expansion path for remaining tasks and graph edges.

If no model is connected or structured output is unavailable, deterministic template fallback is used. This preserves reproducibility while keeping the interface compatible with stronger local or remote LLMs.

## 12. Citation And Source Papers

Reward components and swarm hooks are informed by the following papers:

1. AutoGraph-R1: Enhancing Agentic RAG with Graph-R1 for Complex QA.
  arXiv: https://arxiv.org/abs/2510.15339

2. UniRel: Graph-based Relational Retrieval for LLM Reasoning.
  arXiv: https://arxiv.org/abs/2512.17043

3. DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning.
  EMNLP 2017: https://aclanthology.org/D17-1060/

4. Multi-Hop Knowledge Graph Reasoning with Reward Shaping.
  EMNLP 2018: https://aclanthology.org/D18-1362/

5. Kimi K2.5 (PARL-style multi-agent shaping motivation).
  arXiv: https://arxiv.org/abs/2602.02276

Additional context:

6. MINERVA: Reinforcement Learning for Query Answering over Knowledge Graphs.
  arXiv: https://arxiv.org/abs/1711.05851

## 13. Development And Testing

Run tests from project root:

   pytest -q

Recommended validation after config changes:

1. osint-env demo --agent-mode swarm
2. osint-env eval --episodes 5
3. osint-env benchmark --episodes 5 --name quick_check
4. osint-env leaderboard --top 5

## 14. Scope Boundaries

- This repository supports a low-width swarm baseline and reward-compatible orchestration.
- It does not include a full distributed training stack or asynchronous external worker runtime.
- The architecture keeps those extensions possible without breaking current interfaces.
