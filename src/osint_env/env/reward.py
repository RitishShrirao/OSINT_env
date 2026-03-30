from __future__ import annotations

from osint_env.domain.models import Edge, TaskInstance


def edge_in_truth(edge: Edge, task: TaskInstance) -> bool:
    return any(e.src == edge.src and e.rel == edge.rel and e.dst == edge.dst for e in task.supporting_edges)


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
