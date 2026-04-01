from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from osint_env.domain.models import CanonicalGraph, Edge, TaskInstance


@dataclass(slots=True)
class RewardModel:
    relation_idf: dict[str, float]
    max_relation_idf: float
    hub_penalty: dict[str, float]
    max_hub_penalty: float
    type_priors: dict[tuple[str, str, str], float]


@dataclass(slots=True)
class EdgeRewardBreakdown:
    total: float
    global_accuracy: float
    soft_shaping: float
    efficiency: float
    diversity: float
    relation_informativeness: float
    entity_informativeness: float
    connectivity_gain: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class AnswerRewardBreakdown:
    total: float
    format_reward: float
    correctness: float
    knowledge_carrier: float
    knowledge_indexing: float
    connectivity: float
    graph_f1: float
    efficiency: float
    compactness: float
    relation_informativeness: float
    entity_informativeness: float
    repetition_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def build_reward_model(graph: CanonicalGraph) -> RewardModel:
    relation_freq: Counter[str] = Counter(e.rel for e in graph.edges)
    total_edges = max(1, len(graph.edges))
    relation_idf = {
        rel: math.log((1.0 + total_edges) / (1.0 + freq)) + 1.0 for rel, freq in relation_freq.items()
    }
    max_relation_idf = max(relation_idf.values()) if relation_idf else 1.0

    degree: Counter[str] = Counter()
    for edge in graph.edges:
        degree[edge.src] += 1
        degree[edge.dst] += 1
    hub_penalty = {node_id: math.log(1.0 + deg) for node_id, deg in degree.items()}
    max_hub_penalty = max(hub_penalty.values()) if hub_penalty else 1.0

    type_counts: Counter[tuple[str, str, str]] = Counter()
    rel_counts: Counter[str] = Counter()
    for edge in graph.edges:
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        key = (str(src.node_type.value), edge.rel, str(dst.node_type.value))
        type_counts[key] += 1
        rel_counts[edge.rel] += 1
    type_priors = {
        key: count / max(1, rel_counts[key[1]]) for key, count in type_counts.items()
    }

    return RewardModel(
        relation_idf=relation_idf,
        max_relation_idf=max_relation_idf,
        hub_penalty=hub_penalty,
        max_hub_penalty=max_hub_penalty,
        type_priors=type_priors,
    )


def edge_in_truth(edge: Edge, task: TaskInstance) -> bool:
    return any(e.src == edge.src and e.rel == edge.rel and e.dst == edge.dst for e in task.supporting_edges)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return (num / den) if den else 0.0


def _edge_signature(edge: Edge) -> Counter[str]:
    # Approximate path/edge embedding using relation and endpoint prefixes.
    src_prefix = edge.src.split("_", 1)[0]
    dst_prefix = edge.dst.split("_", 1)[0]
    return Counter({f"rel:{edge.rel}": 2, f"src:{src_prefix}": 1, f"dst:{dst_prefix}": 1})


def _soft_fact_score(edge: Edge, model: RewardModel, graph: CanonicalGraph) -> float:
    if any(e.src == edge.src and e.rel == edge.rel and e.dst == edge.dst for e in graph.edges):
        return 1.0

    src = graph.nodes.get(edge.src)
    dst = graph.nodes.get(edge.dst)
    if src is None or dst is None:
        return 0.0

    type_key = (str(src.node_type.value), edge.rel, str(dst.node_type.value))
    prior = model.type_priors.get(type_key, 0.0)

    # A tiny domain heuristic: alias links are common and worth soft credit even without exact support edge.
    alias_bias = 0.2 if (edge.rel == "alias_of" and edge.src.startswith("alias_") and edge.dst.startswith("user_")) else 0.0
    relation_exists = any(e.rel == edge.rel for e in graph.edges)
    relation_bonus = 0.1 if relation_exists else 0.0
    return max(0.0, min(1.0, 0.1 + (0.65 * prior) + alias_bias + relation_bonus))


def _normalized_relation_info(rel: str, model: RewardModel) -> float:
    idf = model.relation_idf.get(rel, 1.0)
    return idf / max(1e-6, model.max_relation_idf)


def _normalized_entity_info(src: str, dst: str, model: RewardModel) -> float:
    src_h = model.hub_penalty.get(src, 0.0)
    dst_h = model.hub_penalty.get(dst, 0.0)
    mean_hub = (src_h + dst_h) / 2.0
    # UniRel-style preference for low-degree intermediates: lower hub penalty -> higher informativeness.
    return 1.0 - (mean_hub / max(1e-6, model.max_hub_penalty))


def _is_reachable_undirected(edges: list[Edge], src: str, dst: str) -> bool:
    if src == dst:
        return True
    adj: dict[str, set[str]] = {}
    for edge in edges:
        adj.setdefault(edge.src, set()).add(edge.dst)
        adj.setdefault(edge.dst, set()).add(edge.src)
    seen = {src}
    stack = [src]
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, set()):
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _connectivity_gain(edge: Edge, existing_edges: list[Edge]) -> float:
    # Reward edges that bridge disconnected regions and penalize already-connected shortcuts.
    if edge.src == edge.dst:
        return -0.06
    already_connected = _is_reachable_undirected(existing_edges, edge.src, edge.dst)
    if already_connected:
        return -0.03
    return 0.10


def compute_edge_reward(
    edge: Edge,
    task: TaskInstance,
    existing_edges: list[Edge],
    step_count: int,
    model: RewardModel,
    graph: CanonicalGraph,
) -> EdgeRewardBreakdown:
    in_truth = edge_in_truth(edge, task)

    # DeepPath-inspired global accuracy term.
    global_accuracy = 0.85 if in_truth else -0.55

    # D18 reward shaping: R = Rb + (1 - Rb) * f, where f is a soft fact plausibility score.
    base_reward = 1.0 if in_truth else 0.0
    shaped = base_reward + ((1.0 - base_reward) * _soft_fact_score(edge, model, graph))
    soft_shaping = 0.30 * (shaped - 0.5)

    # DeepPath-inspired efficiency term: earlier useful edges are better.
    efficiency = 0.10 * (1.0 / max(1, step_count))

    # DeepPath-inspired diversity term: discourage repeated edge patterns.
    if not existing_edges:
        diversity = 0.08
    else:
        new_sig = _edge_signature(edge)
        avg_similarity = sum(_cosine(new_sig, _edge_signature(e)) for e in existing_edges) / len(existing_edges)
        novelty = 1.0 - avg_similarity
        diversity = 0.14 * (novelty - 0.5)

    # UniRel-style informativeness terms.
    relation_informativeness = 0.12 * (_normalized_relation_info(edge.rel, model) - 0.5)
    entity_informativeness = 0.12 * (_normalized_entity_info(edge.src, edge.dst, model) - 0.5)

    # Additional structural utility shaping for KG construction.
    connectivity_gain = _connectivity_gain(edge, existing_edges)

    total = (
        global_accuracy
        + soft_shaping
        + efficiency
        + diversity
        + relation_informativeness
        + entity_informativeness
        + connectivity_gain
    )
    return EdgeRewardBreakdown(
        total=total,
        global_accuracy=global_accuracy,
        soft_shaping=soft_shaping,
        efficiency=efficiency,
        diversity=diversity,
        relation_informativeness=relation_informativeness,
        entity_informativeness=entity_informativeness,
        connectivity_gain=connectivity_gain,
    )


def _connectivity_ratio(pred_edges: list[Edge], task: TaskInstance) -> float:
    nodes = {e.src for e in task.supporting_edges} | {e.dst for e in task.supporting_edges}
    if len(nodes) <= 1:
        return 1.0

    adj: dict[str, set[str]] = {}
    for edge in pred_edges:
        adj.setdefault(edge.src, set()).add(edge.dst)
        adj.setdefault(edge.dst, set()).add(edge.src)

    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen & nodes) / max(1, len(nodes))


def _knowledge_indexing_recall(task: TaskInstance, tool_outputs: list[dict[str, object]]) -> float:
    gold_terms = {task.answer.lower()}
    for edge in task.supporting_edges:
        gold_terms.add(edge.src.lower())
        gold_terms.add(edge.dst.lower())
        gold_terms.add(edge.rel.lower())

    serialized = json.dumps(tool_outputs).lower()
    covered = sum(1 for term in gold_terms if term and term in serialized)
    return covered / max(1, len(gold_terms))


def _knowledge_carrier_reward(pred_edges: list[Edge], task: TaskInstance) -> float:
    pred = {(e.src, e.rel, e.dst) for e in pred_edges}
    truth = {(e.src, e.rel, e.dst) for e in task.supporting_edges}
    deducible = bool(truth & pred)
    return 0.4 if deducible else -0.2


def _extract_query_entities(question: str) -> set[str]:
    pattern = r"\b(?:alias|user|org|loc|post|thr|thread|event)_[a-zA-Z0-9_]+\b"
    return set(re.findall(pattern, question))


def _max_connected_seed_count(pred_edges: list[Edge], seeds: set[str]) -> int:
    if not seeds:
        return 0
    adj: dict[str, set[str]] = {}
    for edge in pred_edges:
        adj.setdefault(edge.src, set()).add(edge.dst)
        adj.setdefault(edge.dst, set()).add(edge.src)

    best = 1
    for seed in seeds:
        seen = {seed}
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        connected_seed_count = len(seeds & seen)
        best = max(best, connected_seed_count)
    return best


def _unirel_connectivity_score(pred_edges: list[Edge], seeds: set[str]) -> float:
    # UniRel-style discrete connectivity range projected to [-1, 1] for stable weighting.
    n = len(seeds)
    if n <= 1:
        return 0.0

    connected = _max_connected_seed_count(pred_edges, seeds)
    raw = -math.floor(n / 2) + max(0, connected - 1)
    lo = -math.floor(n / 2)
    hi = math.ceil(n / 2) - 1
    if hi <= lo:
        return 0.0
    return ((raw - lo) / (hi - lo)) * 2.0 - 1.0


def _subgraph_relation_informativeness(pred_edges: list[Edge], model: RewardModel | None) -> float:
    if not pred_edges or model is None:
        return 0.0
    avg = sum(_normalized_relation_info(edge.rel, model) for edge in pred_edges) / len(pred_edges)
    return avg - 0.5


def _subgraph_entity_informativeness(pred_edges: list[Edge], model: RewardModel | None) -> float:
    if not pred_edges or model is None:
        return 0.0
    avg = sum(_normalized_entity_info(edge.src, edge.dst, model) for edge in pred_edges) / len(pred_edges)
    return avg - 0.5


def _relation_repetition_ratio(pred_edges: list[Edge]) -> float:
    if len(pred_edges) <= 1:
        return 0.0
    rels = [edge.rel for edge in pred_edges]
    unique = len(set(rels))
    return 1.0 - (unique / len(rels))


def _deducible_answer(proposed_answer: str, task: TaskInstance, pred_edges: list[Edge]) -> bool:
    if proposed_answer != task.answer:
        return False
    truth = {(edge.src, edge.rel, edge.dst) for edge in task.supporting_edges}
    pred = {(edge.src, edge.rel, edge.dst) for edge in pred_edges}
    if truth & pred:
        return True

    seeds = _extract_query_entities(task.question)
    if not seeds:
        return False
    for seed in seeds:
        if _is_reachable_undirected(pred_edges, seed, proposed_answer):
            return True
    return False


def compute_answer_reward(
    proposed_answer: str,
    task: TaskInstance,
    pred_edges: list[Edge],
    tool_outputs: list[dict[str, object]],
    step_count: int,
    model: RewardModel | None = None,
) -> AnswerRewardBreakdown:
    format_reward = 0.15 if proposed_answer else -0.55
    correctness = 1.15 if proposed_answer == task.answer else -1.0

    # AutoGraph-R1 style task utility decomposition.
    knowledge_carrier = 0.50 if _deducible_answer(proposed_answer, task, pred_edges) else -0.25
    knowledge_indexing = 0.45 * _knowledge_indexing_recall(task, tool_outputs)

    # UniRel-style connectivity over seed entities.
    seed_entities = _extract_query_entities(task.question)
    seed_entities.add(task.answer)
    connectivity = 0.30 * _unirel_connectivity_score(pred_edges, seed_entities)

    graph_f1 = 0.55 * compute_graph_f1(pred_edges, task.supporting_edges)
    efficiency = 0.12 * (1.0 / max(1, step_count))

    extra_edges = max(0, len(pred_edges) - len(task.supporting_edges))
    compactness = -0.05 * extra_edges

    relation_informativeness = 0.12 * _subgraph_relation_informativeness(pred_edges, model)
    entity_informativeness = 0.12 * _subgraph_entity_informativeness(pred_edges, model)

    # AutoGraph-R1 repetition control variant used in larger models.
    repetition_penalty = -0.10 * _relation_repetition_ratio(pred_edges)

    total = (
        format_reward
        + correctness
        + knowledge_carrier
        + knowledge_indexing
        + connectivity
        + graph_f1
        + efficiency
        + compactness
        + relation_informativeness
        + entity_informativeness
        + repetition_penalty
    )
    return AnswerRewardBreakdown(
        total=total,
        format_reward=format_reward,
        correctness=correctness,
        knowledge_carrier=knowledge_carrier,
        knowledge_indexing=knowledge_indexing,
        connectivity=connectivity,
        graph_f1=graph_f1,
        efficiency=efficiency,
        compactness=compactness,
        relation_informativeness=relation_informativeness,
        entity_informativeness=entity_informativeness,
        repetition_penalty=repetition_penalty,
    )


def compute_graph_f1(pred_edges: list[Edge], truth_edges: list[Edge]) -> float:
    pred = {(e.src, e.rel, e.dst) for e in pred_edges}
    truth = {(e.src, e.rel, e.dst) for e in truth_edges}
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0
