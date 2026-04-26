# Trace Net: Training Agents for Noisy, Adversarial OSINT

🤖 **Checkpoint:** [Siddeshwar1625/osint-checkpoints-final](https://huggingface.co/Siddeshwar1625/osint-checkpoints-final)

Most agent benchmarks are still too clean.

They assume the world is cooperative, the evidence is tidy, and the shortest path to the answer is also the most obvious one. Real OSINT is the opposite. People hide. Identities splinter across aliases. Threads derail. Posts mislead on purpose. Useful evidence is mixed with decoys, soft contradictions, and deliberate attempts to waste an investigator's time.

That is the motivation behind **Trace Net**.

Trace Net is a synthetic OSINT benchmark environment for tool-using agents that need to search, cross-link, and reason over noisy multi-platform evidence before producing an answer. Instead of rewarding pure prompt cleverness, the environment pushes agents to behave like investigators: retrieve signals, build a working graph, resolve entities, and justify the final node they select.

This repository is not just a dataset and not just a demo app. It is a full benchmark stack:

- a synthetic OSINT environment
- a tunable noise generator
- single-agent and swarm-style execution
- graph-aware reward shaping
- adversarial self-play training
- evaluation, leaderboard, and dashboard export
- a FastAPI/OpenEnv-compatible serving layer for Docker and Hugging Face Spaces

## Why we did not use MetaQA

Earlier iterations explored a MetaQA-style backend, but we deliberately moved away from it for the benchmark we wanted to build.

MetaQA is useful for classic multi-hop reasoning, but it is too clean and too structurally easy for a serious OSINT setting. Once a task becomes mostly "follow a relation chain in a cooperative knowledge base," it stops stress-testing the failure modes we actually care about:

- identity ambiguity
- noisy retrieval
- alias collision
- distractor evidence
- partial observations
- agents being baited into the wrong trail

Trace Net focuses on those harder conditions instead. The goal is not just to see whether a model can traverse hops. The goal is to test whether an agent can survive in an adversarial evidence landscape.

## A noisy world by design

The synthetic dataset is intentionally hostile.

The noise in this repository does not just mean random corruption. It refers to **users actively deflecting agent performance**. Some records act like red herrings. Some identities branch into aliases. Some traces create plausible but misleading routes through the graph. The environment can be tuned so retrieval becomes harder, evidence becomes less direct, and the agent is forced to discriminate signal from manipulation.

This tunability is exposed through environment parameters such as:

- `alias_density`
- `noise_level`
- `red_herring_rate`

Those controls matter because they let the benchmark scale from manageable to punishing without changing the fundamental task structure. You are not switching domains to make the task harder. You are increasing adversarial pressure inside the same OSINT world.

## How the environment works

At the core is a hidden canonical graph of:

- users
- aliases
- organizations
- locations
- posts
- threads
- events

Agents never see this graph directly. They interact through a compact action space:

- `CALL_TOOL`
- `ADD_EDGE`
- `ANSWER`

Every step returns structured observations containing recent tool outputs, the current working-memory graph snapshot, recent action history, and the active task payload. That means agents are evaluated on how they investigate, not just on what final token they emit.

The tool layer exposes search and lookup primitives over synthetic microblog posts, forum threads, profiles, and memory:

- `search_posts`
- `get_post`
- `get_user_posts`
- `get_mentions`
- `search_threads`
- `get_thread`
- `get_user_activity`
- `get_profile`
- `search_people`
- `get_connections`
- `search_memory`
- `search_shared_context`

This turns every episode into a miniature OSINT workflow: gather clues, connect evidence, resist decoys, and only then commit to an answer.

## Multi-agent interaction is a core feature, not a gimmick

One of the strongest ideas in Trace Net is that it treats multi-agent reasoning as an explicit systems problem.

The repository includes both a single-agent runner and a swarm-style orchestrator. In swarm mode, lightweight specialist roles such as **explorer**, **linker**, and **reasoner** coordinate over the same episode. Each role contributes a different kind of progress:

- explorers widen the search frontier
- linkers turn evidence into candidate relations
- reasoners consolidate partial findings into answerable structure

This matters because OSINT naturally decomposes into parallel subproblems. One path follows a person. Another resolves an alias. Another checks whether an event trace is real or planted. A single monolithic agent can do all of that serially, but the benchmark becomes much more interesting when we ask whether a system can split the work, use breadth efficiently, and still converge on the right graph.

Trace Net bakes that into the reward story as well. The swarm runner records spawn count, finished subtasks, critical steps, breadth, and depth. In other words, coordination itself becomes measurable.

## Adversarial self-play is where the benchmark gets dangerous

Trace Net does not stop at evaluation. It includes a scaffold for **adversarial self-play training** built around Hugging Face TRL and the **GRPO** algorithm.

The loop alternates between two roles:

1. a **generator** policy that proposes difficult OSINT tasks
2. an **answerer** policy that tries to solve them

That setup is powerful because it creates pressure from both sides. The generator is rewarded for producing tasks that are valid, grounded, diverse, and hard for the current answerer. The answerer is rewarded using the same environment-native graph-and-answer objectives used during benchmark evaluation.

This is not just hype. The training loop has concrete mathematical logic behind it:

- grouped rollouts for relative comparison
- mean-centered reward baselines through GRPO
- KL-controlled policy updates
- explicit hardness terms against a frozen answerer
- replay validation for generated tasks
- shared-context pressure and swarm diversity terms in `swarm_v2`
- solver-side PARL-style orchestration shaping inspired by **Kimi K2.5**

That means the benchmark can evolve from a static evaluation set into a co-evolving curriculum of adversarial traces. The generator learns how to expose weaknesses. The answerer learns how to survive them.

For OSINT-style agents, that is exactly the kind of training pressure we want.

## Reward design with mathematical grounding

The most important reward story in this repository is the one used during **adversarial self-play training**.

Training uses the **GRPO algorithm** through Hugging Face TRL. That means optimization is driven by grouped rollouts, relative reward comparison inside each group, clipped updates, and KL-regularized policy improvement rather than plain supervised fine-tuning.

In the self-play setting, the generator and solver have different reward functions.

For the **generator**, the training reward is a weighted objective over four core terms:

\[
R_{\text{gen}} =
w_v R_{\text{validity}} +
w_h R_{\text{hardness}} +
w_d R_{\text{diversity}} +
w_c R_{\text{consistency}}
\]

where:

- \(R_{\text{validity}}\) checks that the proposed task is well-formed and bounded
- \(R_{\text{hardness}}\) is higher when the frozen solver fails the generated task
- \(R_{\text{diversity}}\) penalizes near-duplicate generations
- \(R_{\text{consistency}}\) rewards graph-grounded, replayable tasks

In `swarm_v2`, this goes one step further: invalid or non-replayable generations are hard-gated by validation, and the reward then emphasizes replayability, hardness, swarm diversity, and shared-context pressure. This is what keeps the generator from gaming training by emitting flashy but unusable tasks.

For the **solver**, the training reward reuses the environment-native answer reward, but in the self-play pipeline it is explicitly framed as a solver-side objective for adversarial traces. The solver reward is also influenced by the **Kimi K2.5** paper through the project’s PARL-style shaping for multi-agent orchestration. In practice, that means solver training is not only about getting the final answer right, but also about coordinating useful work across the swarm.

The PARL-style orchestration term follows the project’s Kimi-inspired formulation:

\[
r_{\text{PARL}} = r_{\text{perf}} + \lambda_1 r_{\text{parallel}} + \lambda_2 r_{\text{finish}} + r_{\text{latency}}
\]

Therefore the final rewards having the components:

- output format validity and exact correctness
- knowledge-carrier and knowledge-indexing utility
- connectivity and supporting-edge F1 against task support edges,
- efficiency and compactness penalties,
- relation/entity informativeness and repetition control (difficulty-dependent).

This gives the solver-side swarm reward a strong systems flavor: the policy is encouraged to solve the task, but also to do so with effective parallel decomposition instead of brittle serial wandering.

Because training runs under **GRPO**, these rewards are used inside a relative-advantage setting:

- grouped rollouts provide comparison sets
- rewards are contrasted within the group
- KL terms stabilize policy updates
- generator hardness is measured against a frozen solver
- solver improvement is evaluated under the same adversarial pressure that the generator creates

That is the key design choice: the reward is not just scoring answers after the fact. It is shaping a co-evolutionary game between task proposer and task solver.

## Serving, evaluation, and reproducibility

The engineering story is just as compelling as the benchmark design.

The repository provides:

- a `src/` package layout
- CLI commands for `demo`, `eval`, `benchmark`, `leaderboard`, `benchmark-sweep`, `viz`, and `train-self-play`
- artifact outputs for evaluation and baselines
- dashboard export
- a FastAPI server with OpenEnv-style HTTP endpoints
- Docker and Hugging Face Space readiness

This makes Trace Net easy to use in several modes:

- local development
- repeatable benchmarking
- hosted interactive demos
- self-play training runs
- remote evaluation via HTTP

It also means the project is already structured for iteration instead of being locked into a one-off benchmark release.

## Results

The repository already includes reward visualizations and tracking artifacts that make the training story much more concrete.

**Answer reward shaping**

![Answer reward design](https://github.com/RitishShrirao/OSINT_env/blob/main/assets/answer_reward.png?raw=1)

This view highlights that final scoring is not a single accuracy scalar. It combines correctness with graph utility, evidence quality, and efficiency so agents are rewarded for building useful investigative structure.

**Generator reward shaping**

![Generator reward design](https://github.com/RitishShrirao/OSINT_env/blob/main/assets/generator_reward.png?raw=1)

The generator side is where adversarial pressure becomes explicit: validity, hardness, diversity, and consistency work together so the task proposer cannot win by generating nonsense, only by generating hard but replayable traces.

**KL tracking during self-play**

![KL tracking](https://github.com/RitishShrirao/OSINT_env/blob/main/assets/kl.png?raw=1)

KL tracking matters because adversarial training is only useful when updates remain stable. Monitoring KL helps ensure the policies are learning under pressure rather than collapsing into degenerate behavior.

**Checkpoint comparison**
These comparisons have been done after making the queries with the trained generator model
- Finetuned checkpoint: `task_success_rate=0.875`, `avg_reward=0.8996`
- Base model `Qwen/Qwen2.5-0.5B-Instruct`: `task_success_rate=0.0`, `avg_reward=0.5196`
- Delta: `+0.875 success`, `+0.3800 avg reward`

These numbers make the improvement legible at a glance. The finetuned agent moves from zero task success to a strong success rate under the benchmark’s adversarial setting, while also increasing average reward substantially.

## Why Trace Net is exciting

Trace Net is exciting because it pushes agent evaluation closer to the real difficulty profile of OSINT:

- evidence is incomplete
- some actors are deceptive by design
- retrieval can be baited
- graph construction matters
- parallel investigation is valuable
- harder tasks should emerge adversarially, not just be hand-written

A lot of benchmarks ask whether a model can answer. Trace Net asks whether a system can **investigate under pressure**.

That shift is the whole point.

## Quick start

Install locally:

```bash
python -m pip install -e .
```

Run a demo episode:

```bash
osint-env demo --agent-mode swarm --llm-provider mock
```

Run a short benchmark:

```bash
osint-env benchmark --episodes 5 --agent-mode swarm --llm-provider mock --name quick_check
```

Run the release validation gate:

```bash
python scripts/validate_release.py
```

## Final thoughts

Trace Net combines synthetic world-building, adversarial noise, multi-agent coordination, mathematically shaped rewards, and self-play training into one benchmark stack. The result is a far more realistic stress test for OSINT-style agents than clean multi-hop QA can provide.

If the future of agent evaluation is not just "can it answer?" but "can it coordinate, investigate, resist deception, and improve under adversarial pressure?", then Trace Net is pointed in exactly the right direction.
