from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from osint_env.domain.models import CanonicalGraph, Edge, Node, NodeType


_TOPIC_PATTERN = re.compile(r"\[(.*?)\]")


@dataclass(slots=True)
class MetaQATaskRecord:
    question: str
    answers: list[str]
    primary_answer: str
    hop_label: str
    hop_count: int
    split: str
    qtype: str
    topic_entity: str
    supporting_edges: list[Edge]


def _normalize_hop_label(value: str) -> str:
    token = str(value or "").strip().lower().replace(" ", "")
    if token in {"1", "1hop", "1-hop"}:
        return "1-hop"
    if token in {"2", "2hop", "2-hop"}:
        return "2-hop"
    if token in {"3", "3hop", "3-hop"}:
        return "3-hop"
    return ""


def _normalize_split(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in {"train", "dev", "test"}:
        return token
    return ""


def _hop_count(label: str) -> int:
    return int(label.split("-", 1)[0])


def _extract_topic_entity(question: str) -> str:
    match = _TOPIC_PATTERN.search(str(question))
    return match.group(1).strip() if match else ""


def _node_types_for_relation(rel: str) -> tuple[NodeType, NodeType]:
    relation = str(rel or "").strip().lower()
    src_type = NodeType.POST
    if relation in {"directed_by", "written_by", "starred_actors"}:
        return src_type, NodeType.USER
    if relation == "release_year":
        return src_type, NodeType.EVENT
    if relation == "in_language":
        return src_type, NodeType.LOCATION
    if relation in {"has_genre", "has_tags", "has_imdb_votes"}:
        return src_type, NodeType.ORG
    return src_type, NodeType.USER


def _ensure_node(graph: CanonicalGraph, node_id: str, node_type: NodeType) -> None:
    existing = graph.nodes.get(node_id)
    if existing is not None:
        return
    graph.nodes[node_id] = Node(node_id=node_id, node_type=node_type, attrs={"name": node_id})


def _read_non_empty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_kb_line(line: str) -> tuple[str, str, str] | None:
    parts = [part.strip() for part in str(line).split("|", 2)]
    if len(parts) != 3:
        return None
    src, rel, dst = parts
    if not src or not rel or not dst:
        return None
    return src, rel, dst


def _undirected_adjacency(edges: Iterable[Edge]) -> dict[str, list[tuple[str, Edge]]]:
    adj: dict[str, list[tuple[str, Edge]]] = {}
    for edge in edges:
        adj.setdefault(edge.src, []).append((edge.dst, edge))
        adj.setdefault(edge.dst, []).append((edge.src, edge))
    return adj


def _bfs_support_path(
    topic_entity: str,
    answer_candidates: list[str],
    adjacency: dict[str, list[tuple[str, Edge]]],
    max_depth: int,
) -> list[Edge]:
    topic = str(topic_entity or "").strip()
    if not topic or topic not in adjacency:
        return []

    answers = {item.strip() for item in answer_candidates if item.strip()}
    if not answers:
        return []

    queue: deque[tuple[str, list[Edge]]] = deque([(topic, [])])
    visited_depth: dict[str, int] = {topic: 0}

    while queue:
        node, path = queue.popleft()
        depth = len(path)
        if depth > max_depth:
            continue
        if node in answers and path:
            return path
        if depth == max_depth:
            continue
        for neighbor, edge in adjacency.get(node, []):
            next_depth = depth + 1
            best = visited_depth.get(neighbor)
            if best is not None and best <= next_depth:
                continue
            visited_depth[neighbor] = next_depth
            queue.append((neighbor, path + [edge]))
    return []


def _infer_support_edges(
    topic_entity: str,
    answer_candidates: list[str],
    adjacency: dict[str, list[tuple[str, Edge]]],
    hop_count: int,
) -> list[Edge]:
    for limit in (hop_count, hop_count + 1, hop_count + 2, max(4, hop_count + 3)):
        path = _bfs_support_path(topic_entity, answer_candidates, adjacency, max_depth=max(1, limit))
        if path:
            return path
    return []


def infer_metaqa_support_edges(
    graph: CanonicalGraph,
    topic_entity: str,
    answer_candidates: list[str],
    hop_count: int,
) -> list[Edge]:
    adjacency = _undirected_adjacency(graph.edges)
    return _infer_support_edges(
        topic_entity=topic_entity,
        answer_candidates=answer_candidates,
        adjacency=adjacency,
        hop_count=hop_count,
    )


def load_metaqa_dataset(
    root: str | Path,
    kb_path: str | Path | None,
    variant: str,
    hops: list[str],
    splits: list[str],
) -> tuple[CanonicalGraph, list[MetaQATaskRecord]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"MetaQA root not found: {root_path}")

    kb_file = Path(kb_path) if kb_path else root_path / "kb.txt"
    if not kb_file.exists():
        raise FileNotFoundError(f"MetaQA KB file not found: {kb_file}")

    graph = CanonicalGraph()
    seen_edges: set[tuple[str, str, str]] = set()

    for raw_line in _read_non_empty_lines(kb_file):
        row = _parse_kb_line(raw_line)
        if row is None:
            continue
        src, rel, dst = row
        edge_key = (src, rel, dst)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        src_type, dst_type = _node_types_for_relation(rel)
        _ensure_node(graph, src, src_type)
        _ensure_node(graph, dst, dst_type)
        graph.edges.append(Edge(src=src, rel=rel, dst=dst, confidence=1.0))

    hop_labels = [_normalize_hop_label(hop) for hop in hops]
    hop_labels = [hop for hop in hop_labels if hop]
    if not hop_labels:
        hop_labels = ["1-hop", "2-hop", "3-hop"]

    split_labels = [_normalize_split(split) for split in splits]
    split_labels = [split for split in split_labels if split]
    if not split_labels:
        split_labels = ["train", "dev", "test"]

    variant_token = str(variant or "vanilla").strip().lower()
    if variant_token not in {"vanilla", "ntm"}:
        variant_token = "vanilla"

    records: list[MetaQATaskRecord] = []
    for hop in hop_labels:
        hop_dir = root_path / hop
        for split in split_labels:
            qa_path = hop_dir / variant_token / f"qa_{split}.txt"
            if not qa_path.exists():
                continue
            qa_lines = _read_non_empty_lines(qa_path)

            qtype_path = hop_dir / f"qa_{split}_qtype.txt"
            qtypes = _read_non_empty_lines(qtype_path) if qtype_path.exists() else []

            for idx, row in enumerate(qa_lines):
                parts = row.split("\t")
                if len(parts) < 2:
                    continue
                question = parts[0].strip()
                answer_blob = parts[1].strip()
                answers = [item.strip() for item in answer_blob.split("|") if item.strip()]
                if not question or not answers:
                    continue

                topic_entity = _extract_topic_entity(question)
                qtype = qtypes[idx] if idx < len(qtypes) else ""
                records.append(
                    MetaQATaskRecord(
                        question=question,
                        answers=answers,
                        primary_answer=answers[0],
                        hop_label=hop,
                        hop_count=_hop_count(hop),
                        split=split,
                        qtype=qtype,
                        topic_entity=topic_entity,
                        supporting_edges=[],
                    )
                )

    return graph, records
