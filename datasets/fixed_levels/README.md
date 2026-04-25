# Fixed Levels Submission Dataset

This folder contains a fixed three-level OSINT benchmark set built on one shared base graph.

## Files

- `seed_fixed_levels.json`: master fixed seed with an expanded canonical graph and 30 fixed questions.
- `fixed_graph_questions.json`: extracted fixed dataset snapshot for submission packaging.
- `shared_config_fixed_levels.json`: run config used for generation and evaluation.
- `complete_dataset_qwen_generated.json`: full dataset after Qwen (`qwen3:2b` via Ollama) expands the graph.
- `qwen_swarm_eval_fixed_levels.json`: legacy Qwen swarm evaluation summary from the older smaller version of the set.
- `qwen_swarm_benchmark_fixed_levels.json`: legacy benchmark output from the older smaller version of the set.
- `leaderboard_fixed_levels.json`: leaderboard file for this dataset.
- `dashboard_fixed_levels.html`: interactive dashboard generated from the benchmark run.

## Difficulty Design

- Easy: 10 questions. These now use the older hard-style multi-hop traces as the new floor.
- Mid: 10 questions. Each question spans roughly 15-20 supporting nodes.
- High: 10 questions. Each question spans roughly 50 supporting nodes.

All 30 questions are fixed and share the same larger seeded graph.

## Regenerate Artifacts

```bash
source ~/arl/bin/activate
cd /home/ritish/test1
PYTHONPATH=src python scripts/build_fixed_levels_dataset.py \
  --seed-file datasets/fixed_levels/seed_fixed_levels.json \
  --shared-config datasets/fixed_levels/shared_config_fixed_levels.json \
  --output-dir datasets/fixed_levels
```

## Evaluate Qwen Swarm

```bash
source ~/arl/bin/activate
cd /home/ritish/test1
PYTHONPATH=src osint-env eval \
  --config datasets/fixed_levels/shared_config_fixed_levels.json \
  --seed-file datasets/fixed_levels/seed_fixed_levels.json \
  --agent-mode swarm \
  --llm-provider ollama \
  --llm-model qwen3:2b \
  --episodes 15
```

## Benchmark + Dashboard

```bash
source ~/arl/bin/activate
cd /home/ritish/test1
PYTHONPATH=src osint-env benchmark \
  --config datasets/fixed_levels/shared_config_fixed_levels.json \
  --seed-file datasets/fixed_levels/seed_fixed_levels.json \
  --agent-mode swarm \
  --llm-provider ollama \
  --llm-model qwen3:2b \
  --episodes 15 \
  --name fixed_levels_qwen_swarm \
  --leaderboard datasets/fixed_levels/leaderboard_fixed_levels.json \
  --dashboard datasets/fixed_levels/dashboard_fixed_levels.html
```
