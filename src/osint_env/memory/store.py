from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from osint_env.domain.models import Edge


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if t]


@dataclass(slots=True)
class MemoryGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_edge(self, edge: Edge) -> bool:
        key = (edge.src, edge.rel, edge.dst)
        if any((e.src, e.rel, e.dst) == key for e in self.edges):
            return False
        self.edges.append(edge)
        return True

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "nodes_count": len(self.nodes),
            "edges_count": len(self.edges),
            "edges": [{"src": e.src, "rel": e.rel, "dst": e.dst, "confidence": e.confidence} for e in self.edges],
        }


@dataclass(slots=True)
class SemanticMemory:
    docs: list[dict[str, Any]] = field(default_factory=list)

    def add(self, text: str, metadata: dict[str, Any]) -> None:
        self.docs.append({"text": text, "metadata": metadata, "tokens": Counter(_tokenize(text))})

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        q = Counter(_tokenize(query))
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.docs:
            score = self._cosine(q, doc["tokens"])
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, "text": d["text"], "metadata": d["metadata"]} for s, d in scored[:k]]

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return (num / den) if den else 0.0
