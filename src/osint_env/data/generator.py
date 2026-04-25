from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osint_env.data.metaqa import MetaQATaskRecord, infer_metaqa_support_edges, load_metaqa_dataset

from osint_env.domain.models import (
    CanonicalGraph,
    Edge,
    EnvironmentConfig,
    Node,
    NodeType,
    SeedEdgeSpec,
    SeedQuestionSpec,
    TaskInstance,
)

if TYPE_CHECKING:
    from osint_env.llm.interface import LLMClient


@dataclass(slots=True)
class PlatformViews:
    microblog_posts: list[dict]
    forum_threads: list[dict]
    profiles: list[dict]
    alias_lookup: dict[str, str]


def _edge_payload(edge: Edge) -> dict[str, Any]:
    return {
        "src": edge.src,
        "rel": edge.rel,
        "dst": edge.dst,
        "confidence": float(edge.confidence),
    }


def _normalize_swarm_v2_path_edges(path: list[Edge | dict[str, Any]]) -> list[Edge]:
    out: list[Edge] = []
    for row in path:
        if isinstance(row, Edge):
            out.append(Edge(row.src, row.rel, row.dst, float(row.confidence)))
            continue
        if not isinstance(row, dict):
            return []
        src = str(row.get("src", "")).strip()
        rel = str(row.get("rel", "")).strip()
        dst = str(row.get("dst", "")).strip()
        if not src or not rel or not dst:
            return []
        try:
            confidence = float(row.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        out.append(Edge(src=src, rel=rel, dst=dst, confidence=confidence))
    return out


def enumerate_swarm_v2_neighbors(graph: CanonicalGraph, node_id: str) -> list[Edge]:
    edges = [edge for edge in graph.edges if edge.src == node_id]
    edges.sort(key=lambda edge: (edge.src, edge.rel, edge.dst))
    return [Edge(edge.src, edge.rel, edge.dst, float(edge.confidence)) for edge in edges]


def trace_swarm_v2_path(graph: CanonicalGraph, path: list[Edge | dict[str, Any]]) -> list[Edge]:
    edges = _normalize_swarm_v2_path_edges(path)
    if not edges:
        return []

    graph_edges = {(edge.src, edge.rel, edge.dst) for edge in graph.edges}
    for idx, edge in enumerate(edges):
        if (edge.src, edge.rel, edge.dst) not in graph_edges:
            return []
        if idx > 0 and edges[idx - 1].dst != edge.src:
            return []
    return edges


def select_swarm_v2_answer(path_edges: list[Edge]) -> str:
    if not path_edges:
        return ""
    return path_edges[-1].dst


def emit_swarm_v2_question(path_edges: list[Edge]) -> str:
    if not path_edges:
        return ""
    start = path_edges[0].src
    relation_path = " -> ".join(edge.rel for edge in path_edges)
    hops = len(path_edges)
    return (
        f"If you start at {start} and follow the relation path {relation_path}, "
        f"which entity do you reach after {hops} hops?"
    )


def build_swarm_v2_tool_trace(graph: CanonicalGraph, path_edges: list[Edge]) -> list[dict[str, Any]]:
    traced = trace_swarm_v2_path(graph, path_edges)
    if not traced:
        return []

    tool_trace: list[dict[str, Any]] = []
    for idx, edge in enumerate(traced):
        neighbors = enumerate_swarm_v2_neighbors(graph, edge.src)
        tool_trace.append(
            {
                "tool_name": "enumerate_neighbors",
                "args": {
                    "node_id": edge.src,
                    "hop_index": idx,
                    "expected_edge": _edge_payload(edge),
                },
                "output": {
                    "neighbors": [_edge_payload(candidate) for candidate in neighbors],
                },
            }
        )

    tool_trace.append(
        {
            "tool_name": "trace_path",
            "args": {
                "path": [_edge_payload(edge) for edge in traced],
            },
            "output": {
                "path": [_edge_payload(edge) for edge in traced],
            },
        }
    )

    answer = select_swarm_v2_answer(traced)
    tool_trace.append(
        {
            "tool_name": "select_answer",
            "args": {
                "strategy": "path_dst",
            },
            "output": {
                "answer": answer,
            },
        }
    )

    question = emit_swarm_v2_question(traced)
    tool_trace.append(
        {
            "tool_name": "emit_question",
            "args": {
                "style": "relation_path_v1",
            },
            "output": {
                "question": question,
            },
        }
    )
    return tool_trace


def build_swarm_v2_canonical_subgraph(
    graph: CanonicalGraph,
    path_edges: list[Edge],
    max_extra_edges: int = 4,
) -> dict[str, Any]:
    traced = trace_swarm_v2_path(graph, path_edges)
    if not traced:
        return {"nodes": [], "edges": [], "path": []}

    path_nodes = {traced[0].src}
    for edge in traced:
        path_nodes.add(edge.src)
        path_nodes.add(edge.dst)

    path_keys = {(edge.src, edge.rel, edge.dst) for edge in traced}
    extra_edges: list[Edge] = []
    for edge in graph.edges:
        key = (edge.src, edge.rel, edge.dst)
        if key in path_keys:
            continue
        if edge.src in path_nodes or edge.dst in path_nodes:
            extra_edges.append(Edge(edge.src, edge.rel, edge.dst, float(edge.confidence)))
        if len(extra_edges) >= max(0, int(max_extra_edges)):
            break

    subgraph_edges = list(traced) + extra_edges
    subgraph_nodes = sorted({edge.src for edge in subgraph_edges} | {edge.dst for edge in subgraph_edges})
    return {
        "nodes": subgraph_nodes,
        "edges": [_edge_payload(edge) for edge in subgraph_edges],
        "path": [_edge_payload(edge) for edge in traced],
        "answer": select_swarm_v2_answer(traced),
    }


def build_swarm_v2_path_candidates(
    graph: CanonicalGraph,
    rng: random.Random,
    count: int,
    min_hops: int = 2,
    max_hops: int = 4,
) -> list[list[Edge]]:
    if count <= 0:
        return []

    outgoing: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.src, []).append(edge)

    def path_match_count(path: list[Edge], limit: int = 4) -> int:
        if not path:
            return 0
        relations = [edge.rel for edge in path]
        answer = path[-1].dst
        start = path[0].src
        match_count = 0
        stack: list[tuple[str, int, tuple[str, ...]]] = [(start, 0, (start,))]
        while stack:
            node_id, rel_idx, seen_nodes = stack.pop()
            if rel_idx >= len(relations):
                if node_id == answer:
                    match_count += 1
                    if match_count >= limit:
                        return match_count
                continue
            relation = relations[rel_idx]
            for edge in outgoing.get(node_id, []):
                if edge.rel != relation:
                    continue
                if edge.dst in seen_nodes:
                    continue
                stack.append((edge.dst, rel_idx + 1, seen_nodes + (edge.dst,)))
        return match_count

    starts = [node_id for node_id, edges in outgoing.items() if edges]
    if not starts:
        return []

    seen: set[tuple[tuple[str, str, str], ...]] = set()
    candidates: list[list[Edge]] = []
    attempt_budget = max(16, count * 20)
    lower_hops = max(1, int(min_hops))
    upper_hops = max(lower_hops, int(max_hops))

    for _ in range(attempt_budget):
        if len(candidates) >= count:
            break

        current = rng.choice(starts)
        target_hops = rng.randint(lower_hops, upper_hops)
        path: list[Edge] = []
        visited_nodes = {current}

        for _hop in range(target_hops):
            options = [edge for edge in outgoing.get(current, []) if edge.dst not in visited_nodes]
            if not options:
                break
            edge = rng.choice(options)
            path.append(Edge(edge.src, edge.rel, edge.dst, float(edge.confidence)))
            current = edge.dst
            visited_nodes.add(current)

        if len(path) < lower_hops:
            continue
        if path_match_count(path) != 1:
            continue

        key = tuple((edge.src, edge.rel, edge.dst) for edge in path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)

    if candidates:
        return candidates[:count]

    # Fall back to unique 1-hop paths only when the graph is too shallow for multi-hop traces.
    for edge in graph.edges:
        key = ((edge.src, edge.rel, edge.dst),)
        if key in seen:
            continue
        if path_match_count([edge]) != 1:
            continue
        seen.add(key)
        candidates.append([Edge(edge.src, edge.rel, edge.dst, float(edge.confidence))])
        if len(candidates) >= count:
            break
    return candidates[:count]


class DatasetGenerator:
    def __init__(self, config: EnvironmentConfig, llm: LLMClient | None = None):
        self.config = config
        self.rng = random.Random(config.seed)
        self.llm = llm
        self._metaqa_records: list[MetaQATaskRecord] = []

    @staticmethod
    def _edge_key(edge: Edge) -> tuple[str, str, str]:
        return (edge.src, edge.rel, edge.dst)

    def _dataset_mode(self) -> str:
        token = str(getattr(self.config, "dataset_mode", "canonical") or "canonical").strip().lower()
        return "metaqa" if token == "metaqa" else "canonical"

    @staticmethod
    def _metaqa_difficulty(hop_label: str) -> str:
        hop = str(hop_label).strip().lower()
        if hop == "1-hop":
            return "easy"
        if hop == "2-hop":
            return "medium"
        return "hard"

    @staticmethod
    def _infer_node_type(node_id: str) -> NodeType:
        prefix = str(node_id).split("_", 1)[0].lower()
        mapping = {
            "user": NodeType.USER,
            "alias": NodeType.ALIAS,
            "org": NodeType.ORG,
            "loc": NodeType.LOCATION,
            "location": NodeType.LOCATION,
            "post": NodeType.POST,
            "thr": NodeType.THREAD,
            "thread": NodeType.THREAD,
            "event": NodeType.EVENT,
        }
        return mapping.get(prefix, NodeType.USER)

    def _ensure_node(self, graph: CanonicalGraph, node_id: str) -> None:
        if node_id in graph.nodes:
            return
        node_type = self._infer_node_type(node_id)
        attrs: dict[str, Any] = {}
        if node_type == NodeType.USER:
            attrs = {"name": node_id, "org": "Unknown", "location": "Unknown"}
        if node_type == NodeType.ALIAS:
            attrs = {"handle": f"@{node_id}"}
        graph.nodes[node_id] = Node(node_id=node_id, node_type=node_type, attrs=attrs)

    def _add_edge_if_missing(self, graph: CanonicalGraph, edge: Edge) -> None:
        key = self._edge_key(edge)
        if any(self._edge_key(existing) == key for existing in graph.edges):
            return
        self._ensure_node(graph, edge.src)
        self._ensure_node(graph, edge.dst)
        graph.edges.append(edge)

    @staticmethod
    def _extract_json_blob(text: str) -> Any:
        text = str(text).strip()
        if not text:
            return None
        for start, end in (("{", "}"), ("[", "]")):
            left = text.find(start)
            right = text.rfind(end)
            if left >= 0 and right > left:
                snippet = text[left : right + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
        return None

    def _apply_seed_nodes(self, graph: CanonicalGraph) -> None:
        for node_spec in self.config.seeding.seeded_nodes:
            node_type = (
                node_spec.node_type
                if isinstance(node_spec.node_type, NodeType)
                else self._infer_node_type(node_spec.node_id)
            )
            existing = graph.nodes.get(node_spec.node_id)
            attrs = dict(existing.attrs) if existing else {}
            attrs.update(node_spec.attrs)
            graph.nodes[node_spec.node_id] = Node(node_spec.node_id, node_type, attrs)

    def _apply_seed_edges(self, graph: CanonicalGraph) -> None:
        for edge_spec in self.config.seeding.seeded_edges:
            self._add_edge_if_missing(
                graph,
                Edge(
                    src=edge_spec.src,
                    rel=edge_spec.rel,
                    dst=edge_spec.dst,
                    confidence=float(edge_spec.confidence),
                ),
            )

    @staticmethod
    def _normalize_edge_candidates(value: Any) -> list[SeedEdgeSpec]:
        items: list[SeedEdgeSpec] = []
        if not isinstance(value, list):
            return items
        for row in value:
            if not isinstance(row, dict):
                continue
            src = str(row.get("src", "")).strip()
            rel = str(row.get("rel", "")).strip()
            dst = str(row.get("dst", "")).strip()
            if not src or not rel or not dst:
                continue
            try:
                confidence = float(row.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            items.append(SeedEdgeSpec(src=src, rel=rel, dst=dst, confidence=confidence))
        return items

    @staticmethod
    def _split_budget(total: int, parts: int) -> list[int]:
        if total <= 0:
            return []
        slots = max(1, parts)
        base = total // slots
        remainder = total % slots
        chunks = [base + (1 if i < remainder else 0) for i in range(slots)]
        return [chunk for chunk in chunks if chunk > 0]

    @staticmethod
    def _shared_context_blob(graph: CanonicalGraph, node_limit: int = 100, edge_limit: int = 80) -> str:
        payload = {
            "known_nodes": sorted(graph.nodes.keys())[:node_limit],
            "known_edges": [
                {"src": edge.src, "rel": edge.rel, "dst": edge.dst}
                for edge in graph.edges[: min(edge_limit, len(graph.edges))]
            ],
        }
        return json.dumps(payload)

    def _llm_generate_json_with_retry(self, prompt: str) -> Any:
        if self.llm is None:
            return None

        attempts = max(1, int(self.config.seeding.llm_generation_retries))
        for _ in range(attempts):
            try:
                response = self.llm.generate([{"role": "system", "content": prompt}], tools=[])
            except Exception:
                continue
            parsed = self._extract_json_blob(response.content)
            if parsed is not None:
                return parsed
        return None

    def _run_generation_workers(self, prompts: list[str]) -> list[Any]:
        if not prompts:
            return []

        max_workers = max(1, min(self.config.seeding.llm_generation_workers, len(prompts)))
        if not self.config.seeding.llm_generation_parallel or max_workers == 1:
            output: list[Any] = []
            for prompt in prompts:
                parsed = self._llm_generate_json_with_retry(prompt)
                if parsed is not None:
                    output.append(parsed)
            return output

        output = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._llm_generate_json_with_retry, prompt) for prompt in prompts]
            for future in as_completed(futures):
                try:
                    parsed = future.result()
                except Exception:
                    parsed = None
                if parsed is not None:
                    output.append(parsed)
        return output

    def _template_fallback_allowed(self) -> bool:
        if self.llm is None:
            return True
        return bool(self.config.seeding.allow_template_fallback_on_llm_failure)

    def _template_generated_edges(self, graph: CanonicalGraph, budget: int) -> list[Edge]:
        if budget <= 0:
            return []
        users = [n.node_id for n in graph.nodes.values() if n.node_type == NodeType.USER]
        aliases = [n.node_id for n in graph.nodes.values() if n.node_type == NodeType.ALIAS]
        if len(users) < 2:
            return []

        generated: list[Edge] = []
        rels = ["connected_to", "mentions", "co_occurs_with"]
        for _ in range(budget * 3):
            if len(generated) >= budget:
                break
            roll = self.rng.random()
            if aliases and roll < 0.2:
                src = self.rng.choice(aliases)
                dst = self.rng.choice(users)
                rel = "alias_of"
            elif roll < 0.75:
                src, dst = self.rng.sample(users, 2)
                rel = self.rng.choice(rels)
            else:
                src = self.rng.choice(users)
                dst = self.rng.choice([u for u in users if u != src])
                rel = "connected_to"
            generated.append(Edge(src=src, rel=rel, dst=dst, confidence=0.7))
        return generated[:budget]

    def _llm_expand_graph(self, graph: CanonicalGraph, budget: int) -> list[Edge]:
        if budget <= 0:
            return []

        if self.llm is None:
            return self._template_generated_edges(graph, budget)

        shared_context = self._shared_context_blob(graph)
        workers = max(1, min(self.config.seeding.llm_generation_workers, budget))
        chunks = self._split_budget(budget, workers)
        focus_tracks = ["entity_linking", "network_expansion", "org_location", "event_trace"]

        prompts: list[str] = []
        for idx, chunk_budget in enumerate(chunks):
            focus = focus_tracks[idx % len(focus_tracks)]
            prompts.append(
                (
                    "SEED_GRAPH_EXPANSION_AGENT\n"
                    "SHARED_CONTEXT\n"
                    f"{shared_context}\n"
                    f"worker_id: {idx}\n"
                    f"focus: {focus}\n"
                    f"budget: {chunk_budget}\n"
                    "Generate plausible graph edges for OSINT retrieval.\n"
                    "Return STRICT JSON object: {\"edges\": [{\"src\": str, \"rel\": str, \"dst\": str, \"confidence\": float}]}.\n"
                    "Prefer known nodes from SHARED_CONTEXT and avoid duplicates."
                )
            )

        generated: list[Edge] = []
        seen: set[tuple[str, str, str]] = set()
        for payload in self._run_generation_workers(prompts):
            raw_edges: Any = None
            if isinstance(payload, dict):
                raw_edges = payload.get("edges")
            elif isinstance(payload, list):
                raw_edges = payload
            for edge_spec in self._normalize_edge_candidates(raw_edges):
                key = (edge_spec.src, edge_spec.rel, edge_spec.dst)
                if key in seen:
                    continue
                seen.add(key)
                generated.append(Edge(edge_spec.src, edge_spec.rel, edge_spec.dst, float(edge_spec.confidence)))
                if len(generated) >= budget:
                    break
            if len(generated) >= budget:
                break

        if len(generated) < budget:
            residual = budget - len(generated)
            residual_prompt = (
                "SEED_GRAPH_EXPANSION_AGENT\n"
                "SHARED_CONTEXT\n"
                f"{shared_context}\n"
                f"budget: {residual}\n"
                "Generate any remaining high-utility edges.\n"
                "Return STRICT JSON object: {\"edges\": [{\"src\": str, \"rel\": str, \"dst\": str, \"confidence\": float}]}."
            )
            payload = self._llm_generate_json_with_retry(residual_prompt)
            raw_edges: Any = payload.get("edges") if isinstance(payload, dict) else payload
            for edge_spec in self._normalize_edge_candidates(raw_edges):
                key = (edge_spec.src, edge_spec.rel, edge_spec.dst)
                if key in seen:
                    continue
                seen.add(key)
                generated.append(Edge(edge_spec.src, edge_spec.rel, edge_spec.dst, float(edge_spec.confidence)))
                if len(generated) >= budget:
                    break

        if len(generated) < budget and self._template_fallback_allowed():
            for edge in self._template_generated_edges(graph, budget - len(generated)):
                key = (edge.src, edge.rel, edge.dst)
                if key in seen:
                    continue
                seen.add(key)
                generated.append(edge)
                if len(generated) >= budget:
                    break

        return generated[:budget]

    @staticmethod
    def _extract_entity_tokens(question: str) -> list[str]:
        return re.findall(r"\b(?:alias|user|org|loc|post|thr|thread|event)_[a-zA-Z0-9_]+\b", question)

    @staticmethod
    def _normalize_difficulty(value: str, index: int) -> str:
        token = str(value or "").strip().lower()
        if token in {"easy", "e"}:
            return "easy"
        if token in {"mid", "medium", "m"}:
            return "medium"
        if token in {"high", "hard", "h"}:
            return "hard"
        if index < 10:
            return "easy"
        if index < 20:
            return "medium"
        return "hard"

    @staticmethod
    def _task_type_for_difficulty(base_task_type: str, difficulty: str) -> str:
        token = str(base_task_type or "").strip().lower()
        if token and token != "fixed_trace":
            return token
        if difficulty == "easy":
            return "easy_trace"
        if difficulty == "medium":
            return "medium_trace"
        return "hard_trace"

    @staticmethod
    def _grader_for_difficulty(difficulty: str) -> dict[str, Any]:
        return {
            "type": "difficulty_exact_match",
            "answer_type": "node_id",
            "case_sensitive": True,
            "reward_profile": difficulty,
            "logic": {
                "easy": "single_agent_simplified",
                "medium": "reduced_components",
                "hard": "full_reward",
            }.get(difficulty, "full_reward"),
        }

    def _task_metadata(self, index: int, base_task_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(metadata or {})
        difficulty = self._normalize_difficulty(out.get("difficulty", ""), index)
        out["difficulty"] = difficulty
        out.setdefault("grader", self._grader_for_difficulty(difficulty))
        out.setdefault("scenario", self._task_type_for_difficulty(base_task_type, difficulty))
        return out

    def _infer_answer_from_question(self, question: str, graph: CanonicalGraph) -> str:
        entities = self._extract_entity_tokens(question)
        question_l = question.lower()

        alias_tokens = [token for token in entities if token.startswith("alias_")]
        if alias_tokens:
            alias = alias_tokens[0]
            for edge in graph.edges:
                if edge.rel == "alias_of" and edge.src == alias:
                    return edge.dst

        if "connected" in question_l:
            user_tokens = [token for token in entities if token.startswith("user_")]
            if user_tokens:
                source = user_tokens[0]
                for edge in graph.edges:
                    if edge.rel == "connected_to" and edge.src == source:
                        return edge.dst

        if "works at" in question_l:
            for edge in graph.edges:
                if edge.rel != "works_at":
                    continue
                org = graph.nodes.get(edge.dst)
                org_name = str((org.attrs or {}).get("name", "")).lower() if org else ""
                if org_name and org_name in question_l:
                    return edge.src

        return entities[0] if entities else "unknown"

    def _infer_support_edges(self, question: str, answer: str, graph: CanonicalGraph) -> list[Edge]:
        if answer:
            for edge in graph.edges:
                if edge.dst == answer or edge.src == answer:
                    if edge.src in question or edge.dst in question or edge.rel in question.lower():
                        return [edge]

        entities = self._extract_entity_tokens(question)
        for edge in graph.edges:
            if edge.src in entities or edge.dst in entities:
                return [edge]
        return []

    def _seeded_tasks(self, graph: CanonicalGraph) -> list[TaskInstance]:
        tasks: list[TaskInstance] = []
        for idx, question_spec in enumerate(self.config.seeding.seeded_questions):
            answer = question_spec.answer or self._infer_answer_from_question(question_spec.question, graph)
            metadata = self._task_metadata(idx, question_spec.task_type, dict(question_spec.metadata))
            difficulty = str(metadata.get("difficulty", "hard"))
            if question_spec.supporting_edges:
                support = [
                    Edge(src=e.src, rel=e.rel, dst=e.dst, confidence=float(e.confidence))
                    for e in question_spec.supporting_edges
                ]
            else:
                support = self._infer_support_edges(question_spec.question, answer, graph)

            tasks.append(
                TaskInstance(
                    task_id=f"seed_task_{idx}",
                    task_type=self._task_type_for_difficulty(question_spec.task_type, difficulty),
                    question=question_spec.question,
                    answer=answer,
                    supporting_edges=support,
                    metadata=metadata,
                )
            )
        return tasks

    def _template_tasks(self, graph: CanonicalGraph, count: int, start_idx: int = 0) -> list[TaskInstance]:
        alias_edges = [e for e in graph.edges if e.rel == "alias_of"]
        conn_edges = [e for e in graph.edges if e.rel == "connected_to"]
        work_edges = [e for e in graph.edges if e.rel == "works_at"]
        tasks: list[TaskInstance] = []

        for i in range(count):
            mode = self.rng.choice(["identity_resolution", "network_discovery", "event_tracing"])
            if mode == "identity_resolution" and alias_edges:
                edge = self.rng.choice(alias_edges)
                q = f"Which canonical user owns alias {edge.src}?"
                a = edge.dst
                support = [edge]
            elif mode == "network_discovery" and conn_edges:
                edge = self.rng.choice(conn_edges)
                q = f"Who is connected to {edge.src}?"
                a = edge.dst
                support = [edge]
            else:
                edge = self.rng.choice(work_edges)
                org_node = graph.nodes.get(edge.dst)
                org_name = (org_node.attrs or {}).get("name", edge.dst) if org_node else edge.dst
                q = f"Which user works at {org_name}?"
                a = edge.src
                support = [edge]
            tasks.append(
                TaskInstance(
                    task_id=f"task_{start_idx + i}",
                    task_type=mode,
                    question=q,
                    answer=a,
                    supporting_edges=support,
                    metadata=self._task_metadata(start_idx + i, mode),
                )
            )
        return tasks

    def _llm_generated_tasks(self, graph: CanonicalGraph, count: int, start_idx: int) -> list[TaskInstance]:
        if count <= 0:
            return []
        if self.llm is None:
            return self._template_tasks(graph, count=count, start_idx=start_idx)

        candidate_edges = [
            {"src": edge.src, "rel": edge.rel, "dst": edge.dst}
            for edge in graph.edges
            if edge.rel in {"alias_of", "connected_to", "works_at"}
        ][:60]
        shared_context = json.dumps(
            {
                "known_nodes": sorted(graph.nodes.keys())[:100],
                "edge_sample": candidate_edges,
            }
        )
        workers = max(1, min(self.config.seeding.llm_generation_workers, count))
        chunks = self._split_budget(count, workers)
        focus_tracks = ["identity_resolution", "network_discovery", "event_tracing", "deanonymization"]

        prompts: list[str] = []
        for idx, chunk_budget in enumerate(chunks):
            focus = focus_tracks[idx % len(focus_tracks)]
            prompts.append(
                (
                    "SEED_TASK_EXPANSION_AGENT\n"
                    "SHARED_CONTEXT\n"
                    f"{shared_context}\n"
                    f"worker_id: {idx}\n"
                    f"focus: {focus}\n"
                    f"task_budget: {chunk_budget}\n"
                    "Generate OSINT QA tasks with answers and support edges.\n"
                    "Return STRICT JSON object: {\"tasks\": [{\"task_type\": str, \"question\": str, \"answer\": str, \"supporting_edges\": [{\"src\": str, \"rel\": str, \"dst\": str, \"confidence\": float}]}]}."
                )
            )

        llm_tasks: list[TaskInstance] = []
        seen_questions: set[str] = set()
        for payload in self._run_generation_workers(prompts):
            raw_tasks: Any = None
            if isinstance(payload, dict):
                raw_tasks = payload.get("tasks")
            elif isinstance(payload, list):
                raw_tasks = payload
            if not isinstance(raw_tasks, list):
                continue

            for row in raw_tasks:
                if not isinstance(row, dict):
                    continue
                question = str(row.get("question", "")).strip()
                if not question:
                    continue
                key = question.lower()
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                answer = str(row.get("answer", "")).strip() or self._infer_answer_from_question(question, graph)
                task_type = str(row.get("task_type", "llm_generated")).strip() or "llm_generated"
                support_specs = self._normalize_edge_candidates(row.get("supporting_edges"))
                if support_specs:
                    support = [Edge(e.src, e.rel, e.dst, e.confidence) for e in support_specs]
                else:
                    support = self._infer_support_edges(question, answer, graph)
                llm_tasks.append(
                    TaskInstance(
                        task_id=f"task_{start_idx + len(llm_tasks)}",
                        task_type=task_type,
                        question=question,
                        answer=answer,
                        supporting_edges=support,
                        metadata=self._task_metadata(
                            start_idx + len(llm_tasks),
                            task_type,
                            {"generated_by": "llm", "shared_context": True},
                        ),
                    )
                )
                if len(llm_tasks) >= count:
                    break
            if len(llm_tasks) >= count:
                break

        if len(llm_tasks) < count:
            residual = count - len(llm_tasks)
            residual_prompt = (
                "SEED_TASK_EXPANSION_AGENT\n"
                "SHARED_CONTEXT\n"
                f"{shared_context}\n"
                f"task_budget: {residual}\n"
                "Generate additional tasks not already present in SHARED_CONTEXT.\n"
                "Return STRICT JSON object: {\"tasks\": [{\"task_type\": str, \"question\": str, \"answer\": str, \"supporting_edges\": [{\"src\": str, \"rel\": str, \"dst\": str, \"confidence\": float}]}]}."
            )
            payload = self._llm_generate_json_with_retry(residual_prompt)
            raw_tasks: Any = payload.get("tasks") if isinstance(payload, dict) else payload
            if isinstance(raw_tasks, list):
                for row in raw_tasks:
                    if not isinstance(row, dict):
                        continue
                    question = str(row.get("question", "")).strip()
                    if not question:
                        continue
                    key = question.lower()
                    if key in seen_questions:
                        continue
                    seen_questions.add(key)
                    answer = str(row.get("answer", "")).strip() or self._infer_answer_from_question(question, graph)
                    task_type = str(row.get("task_type", "llm_generated")).strip() or "llm_generated"
                    support_specs = self._normalize_edge_candidates(row.get("supporting_edges"))
                    if support_specs:
                        support = [Edge(e.src, e.rel, e.dst, e.confidence) for e in support_specs]
                    else:
                        support = self._infer_support_edges(question, answer, graph)
                    llm_tasks.append(
                        TaskInstance(
                            task_id=f"task_{start_idx + len(llm_tasks)}",
                            task_type=task_type,
                            question=question,
                            answer=answer,
                            supporting_edges=support,
                            metadata=self._task_metadata(
                                start_idx + len(llm_tasks),
                                task_type,
                                {"generated_by": "llm", "shared_context": True},
                            ),
                        )
                    )
                    if len(llm_tasks) >= count:
                        break

        if len(llm_tasks) < count and self._template_fallback_allowed():
            llm_tasks.extend(
                self._template_tasks(
                    graph,
                    count=count - len(llm_tasks),
                    start_idx=start_idx + len(llm_tasks),
                )
            )
        return llm_tasks[:count]

    def _metaqa_selected_records(self, count: int) -> list[MetaQATaskRecord]:
        records = list(self._metaqa_records)
        if not records:
            return []
        if count <= 0 or len(records) <= count:
            return records

        grouped: dict[str, list[MetaQATaskRecord]] = {}
        for record in records:
            grouped.setdefault(record.hop_label, []).append(record)

        hop_keys = sorted(grouped.keys())
        if not hop_keys:
            return records[:count]

        selected: list[MetaQATaskRecord] = []
        leftovers: list[MetaQATaskRecord] = []
        per_hop = max(1, count // len(hop_keys))

        for hop in hop_keys:
            bucket = list(grouped[hop])
            self.rng.shuffle(bucket)
            take = min(len(bucket), per_hop)
            selected.extend(bucket[:take])
            leftovers.extend(bucket[take:])

        if len(selected) < count:
            self.rng.shuffle(leftovers)
            selected.extend(leftovers[: count - len(selected)])

        return selected[:count]

    def _metaqa_tasks(self, graph: CanonicalGraph, count: int) -> list[TaskInstance]:
        records = self._metaqa_selected_records(count)
        tasks: list[TaskInstance] = []
        for idx, record in enumerate(records):
            difficulty = self._metaqa_difficulty(record.hop_label)
            support_edges = list(record.supporting_edges)
            if not support_edges:
                support_edges = infer_metaqa_support_edges(
                    graph=graph,
                    topic_entity=record.topic_entity,
                    answer_candidates=record.answers,
                    hop_count=record.hop_count,
                )
            metadata = {
                "difficulty": difficulty,
                "hop": record.hop_label,
                "split": record.split,
                "source": "metaqa",
                "dataset_mode": "metaqa",
                "qtype": record.qtype,
                "topic_entity": record.topic_entity,
                "all_answers": list(record.answers),
                "grader": {
                    "type": "metaqa_exact_match",
                    "answer_type": "entity_name",
                    "case_sensitive": True,
                    "reward_profile": difficulty,
                    "logic": "hop_trace",
                },
                "scenario": f"metaqa_{record.hop_label}",
            }
            task_type = f"metaqa_{record.hop_label}"
            tasks.append(
                TaskInstance(
                    task_id=f"metaqa_{record.hop_label}_{record.split}_{idx}",
                    task_type=task_type,
                    question=record.question,
                    answer=record.primary_answer,
                    supporting_edges=support_edges,
                    metadata=metadata,
                )
            )
        return tasks

    def _build_platform_views_metaqa(self, graph: CanonicalGraph) -> PlatformViews:
        node_names = {
            node_id: str((node.attrs or {}).get("name") or node_id)
            for node_id, node in graph.nodes.items()
        }

        microblog_posts: list[dict] = []
        for idx, edge in enumerate(graph.edges):
            microblog_posts.append(
                {
                    "post_id": f"post_metaqa_{idx}",
                    "user_id": edge.src,
                    "canonical_user": edge.src,
                    "text": f"{edge.src} {edge.rel} {edge.dst}",
                    "references": [edge.src, edge.dst],
                    "reference_names": [node_names.get(edge.src, edge.src), node_names.get(edge.dst, edge.dst)],
                    "mentions": [edge.dst],
                    "timestamp": 100000 + idx,
                }
            )

        relation_groups: dict[str, list[Edge]] = {}
        for edge in graph.edges:
            relation_groups.setdefault(edge.rel, []).append(edge)

        forum_threads: list[dict] = []
        for idx, rel in enumerate(sorted(relation_groups.keys())[:200]):
            group = relation_groups.get(rel, [])[:10]
            forum_threads.append(
                {
                    "thread_id": f"thr_metaqa_{idx}",
                    "topic": rel,
                    "author_id": group[0].src if group else "metaqa",
                    "comments": [
                        {
                            "user_id": edge.src,
                            "text": f"{edge.src} {edge.rel} {edge.dst}",
                        }
                        for edge in group
                    ],
                    "references": [edge.dst for edge in group],
                    "discusses": [edge.dst for edge in group],
                }
            )

        neighbors: dict[str, set[str]] = {}
        for edge in graph.edges:
            neighbors.setdefault(edge.src, set()).add(edge.dst)
            neighbors.setdefault(edge.dst, set()).add(edge.src)

        profiles: list[dict] = []
        for node_id in sorted(graph.nodes.keys()):
            node = graph.nodes[node_id]
            profiles.append(
                {
                    "user_id": node_id,
                    "name": str((node.attrs or {}).get("name") or node_id),
                    "org": str(node.node_type.value),
                    "org_id": str(node.node_type.value),
                    "location": "metaqa",
                    "location_id": "metaqa",
                    "alias_ids": [],
                    "connections": sorted(neighbors.get(node_id, set()))[:8],
                    "work_history": [str(node.node_type.value)],
                }
            )

        return PlatformViews(
            microblog_posts=microblog_posts,
            forum_threads=forum_threads,
            profiles=profiles,
            alias_lookup={},
        )

    def build_canonical_graph(self) -> CanonicalGraph:
        if self._dataset_mode() == "metaqa":
            root = Path(self.config.metaqa_root)
            kb_path = Path(self.config.metaqa_kb_path) if str(self.config.metaqa_kb_path).strip() else None
            graph, records = load_metaqa_dataset(
                root=root,
                kb_path=kb_path,
                variant=self.config.metaqa_variant,
                hops=list(self.config.metaqa_hops),
                splits=list(self.config.metaqa_splits),
            )
            self._metaqa_records = records
            self._apply_seed_nodes(graph)
            self._apply_seed_edges(graph)
            return graph

        graph = CanonicalGraph()
        orgs = ["Apex Dynamics", "Helios Labs", "Northbridge"]
        locations = ["Bengaluru", "Pune", "Hyderabad", "Delhi"]

        for i in range(self.config.n_users):
            uid = f"user_{i}"
            org = self.rng.choice(orgs)
            loc = self.rng.choice(locations)
            graph.nodes[uid] = Node(uid, NodeType.USER, {"name": f"Person {i}", "org": org, "location": loc})
            org_id = f"org_{org.lower().replace(' ', '_')}"
            loc_id = f"loc_{loc.lower()}"
            graph.nodes.setdefault(org_id, Node(org_id, NodeType.ORG, {"name": org}))
            graph.nodes.setdefault(loc_id, Node(loc_id, NodeType.LOCATION, {"name": loc}))
            graph.edges.append(Edge(uid, "works_at", org_id))
            graph.edges.append(Edge(uid, "located_in", loc_id))

            if self.rng.random() < self.config.alias_density:
                alias = f"alias_{i}_{self.rng.randint(100,999)}"
                graph.nodes[alias] = Node(alias, NodeType.ALIAS, {"handle": f"@{alias}"})
                graph.edges.append(Edge(alias, "alias_of", uid))

        users = [n for n in graph.nodes.values() if n.node_type == NodeType.USER]
        for _ in range(max(1, self.config.n_users // 2)):
            a, b = self.rng.sample(users, 2)
            graph.edges.append(Edge(a.node_id, "connected_to", b.node_id, confidence=0.8))

        self._apply_seed_nodes(graph)
        self._apply_seed_edges(graph)

        if self.config.seeding.llm_generate_remaining_graph:
            llm_edges = self._llm_expand_graph(graph, self.config.seeding.llm_generated_edge_budget)
            for edge in llm_edges:
                self._add_edge_if_missing(graph, edge)
        return graph

    def build_platform_views(self, graph: CanonicalGraph) -> PlatformViews:
        if self._dataset_mode() == "metaqa":
            return self._build_platform_views_metaqa(graph)

        users = [n for n in graph.nodes.values() if n.node_type == NodeType.USER]
        aliases = [n for n in graph.nodes.values() if n.node_type == NodeType.ALIAS]
        alias_owner = {e.src: e.dst for e in graph.edges if e.rel == "alias_of"}
        user_aliases: dict[str, list[str]] = {}
        for alias_id, user_id in alias_owner.items():
            user_aliases.setdefault(user_id, []).append(alias_id)
        node_names = {
            node_id: str((node.attrs or {}).get("name") or (node.attrs or {}).get("handle") or node_id)
            for node_id, node in graph.nodes.items()
        }

        microblog_posts: list[dict] = []
        for i, user in enumerate(users):
            poster = user.node_id
            if aliases and self.rng.random() < 0.45:
                candidate = self.rng.choice(aliases).node_id
                poster = candidate
            text = f"Update {i} from {user.attrs['org']} #{user.attrs['location'].lower()}"
            if self.rng.random() < self.config.noise_level:
                text = f"Rumor: {text} maybe fake"
            microblog_posts.append(
                {
                    "post_id": f"post_{i}",
                    "user_id": poster,
                    "canonical_user": alias_owner.get(poster, user.node_id),
                    "text": text,
                    "references": [],
                    "reference_names": [],
                    "mentions": [f"user_{self.rng.randint(0, self.config.n_users - 1)}"],
                    "timestamp": 1000 + i,
                }
            )

        authored_posts: dict[str, str] = {}
        post_references: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.rel == "authored_post":
                authored_posts[edge.dst] = edge.src
            elif edge.rel == "references" and edge.src.startswith("post_"):
                post_references.setdefault(edge.src, []).append(edge.dst)

        for post_id, author_id in authored_posts.items():
            refs = post_references.get(post_id, [])
            ref_names = [node_names.get(ref, ref) for ref in refs]
            author_label = node_names.get(author_id, author_id)
            text_parts = [f"{post_id} update from {author_label}"]
            if ref_names:
                text_parts.append("references " + ", ".join(ref_names))
            if refs:
                text_parts.append("ids " + ", ".join(refs))
            post_payload = {
                "post_id": post_id,
                "user_id": author_id,
                "canonical_user": alias_owner.get(author_id, author_id),
                "text": ". ".join(text_parts),
                "references": refs,
                "reference_names": ref_names,
                "mentions": [],
                "timestamp": 5000 + len(microblog_posts),
            }
            existing_idx = next((idx for idx, row in enumerate(microblog_posts) if row["post_id"] == post_id), None)
            if existing_idx is None:
                microblog_posts.append(post_payload)
            else:
                microblog_posts[existing_idx] = post_payload

        forum_threads: list[dict] = []
        for i in range(max(8, self.config.n_users // 3)):
            author = self.rng.choice(users).node_id
            forum_threads.append(
                {
                    "thread_id": f"thr_{i}",
                    "topic": self.rng.choice(["security", "startup", "ai", "infra"]),
                    "author_id": author,
                    "comments": [
                        {"user_id": self.rng.choice(users).node_id, "text": "Following this."},
                        {"user_id": self.rng.choice(users).node_id, "text": "Interesting link."},
                    ],
                    "references": [],
                    "discusses": [],
                }
            )

        authored_threads: dict[str, str] = {}
        thread_refs: dict[str, list[str]] = {}
        thread_discusses: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.rel == "authored_thread":
                authored_threads[edge.dst] = edge.src
            elif edge.rel == "references" and edge.src.startswith(("thr_", "thread_")):
                thread_refs.setdefault(edge.src, []).append(edge.dst)
            elif edge.rel == "discusses" and edge.src.startswith(("thr_", "thread_")):
                thread_discusses.setdefault(edge.src, []).append(edge.dst)

        for thread_id, author_id in authored_threads.items():
            node = graph.nodes.get(thread_id)
            refs = thread_refs.get(thread_id, [])
            discussed = thread_discusses.get(thread_id, [])
            comments = []
            for ref in refs:
                comments.append({"user_id": author_id, "text": f"Reference: {node_names.get(ref, ref)} ({ref})"})
            for item in discussed:
                comments.append({"user_id": author_id, "text": f"Discusses: {node_names.get(item, item)} ({item})"})
            thread_payload = {
                "thread_id": thread_id,
                "topic": str((node.attrs or {}).get("topic", "seeded")) if node else "seeded",
                "author_id": author_id,
                "title": node_names.get(thread_id, thread_id),
                "comments": comments,
                "references": refs,
                "discusses": discussed,
            }
            existing_idx = next((idx for idx, row in enumerate(forum_threads) if row["thread_id"] == thread_id), None)
            if existing_idx is None:
                forum_threads.append(thread_payload)
            else:
                forum_threads[existing_idx] = thread_payload

        profiles: list[dict] = []
        for user in users:
            conns = [e.dst for e in graph.edges if e.src == user.node_id and e.rel == "connected_to"][:5]
            org_id = next((e.dst for e in graph.edges if e.src == user.node_id and e.rel == "works_at"), "")
            location_id = next((e.dst for e in graph.edges if e.src == user.node_id and e.rel == "located_in"), "")
            profiles.append(
                {
                    "user_id": user.node_id,
                    "name": user.attrs["name"],
                    "org": user.attrs["org"],
                    "org_id": org_id,
                    "location": user.attrs["location"],
                    "location_id": location_id,
                    "alias_ids": sorted(user_aliases.get(user.node_id, [])),
                    "connections": conns,
                    "work_history": [user.attrs["org"]],
                }
            )

        for i in range(int(len(users) * self.config.red_herring_rate)):
            profiles.append(
                {
                    "user_id": f"noise_{i}",
                    "name": f"P{self.rng.randint(100,999)}",
                    "org": self.rng.choice(["Stealth Co", "Unknown Ventures"]),
                    "org_id": "",
                    "location": self.rng.choice(["Remote", "Unknown"]),
                    "location_id": "",
                    "alias_ids": [],
                    "connections": [],
                    "work_history": [],
                }
            )
        return PlatformViews(microblog_posts, forum_threads, profiles, alias_lookup=alias_owner)

    def generate_tasks(self, graph: CanonicalGraph, views: PlatformViews, count: int = 12) -> list[TaskInstance]:
        if self._dataset_mode() == "metaqa":
            metaqa_tasks = self._metaqa_tasks(graph=graph, count=max(1, count))
            if metaqa_tasks:
                return metaqa_tasks

        tasks = self._seeded_tasks(graph)
        target_count = max(1, count, len(tasks))

        llm_budget = min(
            max(0, self.config.seeding.llm_generated_task_budget),
            max(0, target_count - len(tasks)),
        )
        if self.config.seeding.llm_generate_remaining_tasks and llm_budget > 0:
            tasks.extend(self._llm_generated_tasks(graph, count=llm_budget, start_idx=len(tasks)))

        if len(tasks) < target_count and self._template_fallback_allowed():
            tasks.extend(self._template_tasks(graph, count=target_count - len(tasks), start_idx=len(tasks)))

        if not tasks:
            tasks.extend(self._template_tasks(graph, count=target_count, start_idx=0))

        return tasks[:target_count]
