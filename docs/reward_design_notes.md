# Reward Design Notes

This environment uses a composite reward that adapts ideas from:

- AutoGraph-R1 (arXiv:2510.15339)
- UniRel (arXiv:2512.17043)
- DeepPath (EMNLP 2017, D17-1060)
- Multi-Hop KG Reasoning with Reward Shaping (EMNLP 2018, D18-1362)
- Kimi K2.5 (arXiv:2602.02276) for PARL-style swarm auxiliary shaping

Additional related context consulted:

- MINERVA (arXiv:1711.05851) for query-conditioned walk-style reasoning over KG paths.

## Components in this Branch

The implementation follows a staged reward design:

1. edge-level rewards during graph construction (`ADD_EDGE`)
2. answer-level rewards for retrieval usefulness and final task utility (`ANSWER`)
3. evaluation-level composite leaderboard score for benchmark ranking

### 1) Edge addition reward

For each `ADD_EDGE`, the reward combines:

- Global accuracy term (DeepPath):
  - $r_{global} = +1$ if a candidate edge is correct, else $-1$ (scaled in code for stability).
- Soft shaping term (D18 reward shaping):
  - $R = R_b + (1 - R_b) f(s, r, o)$, where $f$ is a soft fact plausibility score.
  - In code, $f$ is approximated by relation/type priors plus small domain priors.
- Efficiency term (DeepPath):
  - $r_{efficiency} \propto 1 / \text{step\_count}$.
- Diversity term (DeepPath):
  - novelty from cosine dissimilarity of edge signatures; repeated patterns are down-weighted.
- Relation/entity informativeness (UniRel):
  - relation rarity via normalized IDF of relation labels,
  - entity informativeness via inverse hub-penalty.
- Connectivity gain term:
  - rewards bridge edges that connect previously disconnected graph regions.

### 2) Final answer reward

For `ANSWER`, the reward combines:

- format validity,
- answer correctness,
- knowledge-carrying utility (AutoGraph-R1 style):
  - $R_C(q, y, G) = \mathbb{{I}}[\text{{deducible}}(q, y \mid G)]$.
- knowledge-indexing utility (AutoGraph-R1 style):
  - $R_I(q, D_{{gold}}, G) = |Top\text{{-}}k(G,q) \cap D_{{gold}}| / |D_{{gold}}|$,
  - approximated in this environment with evidence recall over tool outputs.
- connectivity (UniRel style):
  - discrete connectivity reward over extracted seed entities, normalized for stable mixing.
- graph F1 against supporting edges,
- compactness penalty for unnecessary extra edges,
- efficiency bonus,
- relation/entity informativeness for the constructed subgraph,
- repetition penalty to discourage redundant relation generation patterns.

UniRel-style aggregate view represented in this branch:

$$
R(a) \approx R_{{fmt}} + R_{{con}} + w_1 R_{{ent}} + w_2 R_{{rel}} + \text{{task utility terms}}
$$

with task utility terms coming from AutoGraph-inspired $R_C$ and $R_I$ components.

## Telemetry

Per-step component rewards are aggregated into `info["reward_components"]`, enabling:

- richer benchmark summaries,
- leaderboard ranking by composite utility,
- visual diagnostics in dashboard exports.

Evaluation also computes derived retrieval and structural utility signals used in leaderboard ranking.

## Future Multi-Agent Notes

This branch now includes a low-width swarm baseline orchestrator that adds PARL-style auxiliary shaping on top of the core edge and answer rewards.

The helper implementation is in:

- `src/osint_env/env/spawn_reward_hooks.py`

It follows the Kimi K2.5 style decomposition:

- $r_{{PARL}}(x,y) = r_{{perf}}(x,y) + \lambda_1 r_{{parallel}} + \lambda_2 r_{{finish}}$,
- optional critical-steps shaping for latency-sensitive training,
- optional annealing of $\lambda_1, \lambda_2$ toward zero,
- optional breadth/depth shaping hooks for future branch integration.

The expanded project-level walkthrough is in `README.md` under "Reward Design (Integrated Notes)".
