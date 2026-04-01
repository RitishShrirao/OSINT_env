from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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


class DatasetGenerator:
    def __init__(self, config: EnvironmentConfig, llm: LLMClient | None = None):
        self.config = config
        self.rng = random.Random(config.seed)
        self.llm = llm

    @staticmethod
    def _edge_key(edge: Edge) -> tuple[str, str, str]:
        return (edge.src, edge.rel, edge.dst)

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

        sample_edges = [
            {"src": edge.src, "rel": edge.rel, "dst": edge.dst}
            for edge in graph.edges[: min(40, len(graph.edges))]
        ]
        sample_nodes = sorted(graph.nodes.keys())[:80]
        prompt = (
            "SEED_GRAPH_EXPANSION\n"
            "Generate additional plausible graph edges to improve retrieval for OSINT tasks.\n"
            "Return STRICT JSON object: {\"edges\": [{\"src\": str, \"rel\": str, \"dst\": str, \"confidence\": float}]}.\n"
            "Use only known node ids when possible. Avoid duplicates.\n"
            f"Budget: {budget}\n"
            f"Known nodes: {json.dumps(sample_nodes)}\n"
            f"Known edges sample: {json.dumps(sample_edges)}"
        )
        response = self.llm.generate([{"role": "system", "content": prompt}], tools=[])
        parsed = self._extract_json_blob(response.content)
        if isinstance(parsed, dict):
            edges = self._normalize_edge_candidates(parsed.get("edges"))
            if edges:
                return [
                    Edge(src=e.src, rel=e.rel, dst=e.dst, confidence=float(e.confidence))
                    for e in edges[:budget]
                ]
        return self._template_generated_edges(graph, budget)

    @staticmethod
    def _extract_entity_tokens(question: str) -> list[str]:
        return re.findall(r"\b(?:alias|user|org|loc|post|thr|thread|event)_[a-zA-Z0-9_]+\b", question)

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
                    task_type=question_spec.task_type,
                    question=question_spec.question,
                    answer=answer,
                    supporting_edges=support,
                    metadata=dict(question_spec.metadata),
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
        prompt = (
            "SEED_TASK_EXPANSION\n"
            "Generate additional OSINT QA tasks from this graph sample.\n"
            "Return STRICT JSON object: {\"tasks\": [{\"task_type\": str, \"question\": str, \"answer\": str, \"supporting_edges\": [{\"src\": str, \"rel\": str, \"dst\": str}]}]}.\n"
            f"Task budget: {count}\n"
            f"Edge sample: {json.dumps(candidate_edges)}"
        )
        response = self.llm.generate([{"role": "system", "content": prompt}], tools=[])
        parsed = self._extract_json_blob(response.content)

        llm_tasks: list[TaskInstance] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("tasks"), list):
            for i, row in enumerate(parsed["tasks"]):
                if not isinstance(row, dict):
                    continue
                question = str(row.get("question", "")).strip()
                if not question:
                    continue
                answer = str(row.get("answer", "")).strip() or self._infer_answer_from_question(question, graph)
                task_type = str(row.get("task_type", "llm_generated")).strip() or "llm_generated"
                support_specs = self._normalize_edge_candidates(row.get("supporting_edges"))
                if support_specs:
                    support = [Edge(e.src, e.rel, e.dst, e.confidence) for e in support_specs]
                else:
                    support = self._infer_support_edges(question, answer, graph)
                llm_tasks.append(
                    TaskInstance(
                        task_id=f"task_{start_idx + i}",
                        task_type=task_type,
                        question=question,
                        answer=answer,
                        supporting_edges=support,
                        metadata={"generated_by": "llm"},
                    )
                )
                if len(llm_tasks) >= count:
                    break

        if len(llm_tasks) < count:
            llm_tasks.extend(
                self._template_tasks(
                    graph,
                    count=count - len(llm_tasks),
                    start_idx=start_idx + len(llm_tasks),
                )
            )
        return llm_tasks[:count]

    def build_canonical_graph(self) -> CanonicalGraph:
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
        users = [n for n in graph.nodes.values() if n.node_type == NodeType.USER]
        aliases = [n for n in graph.nodes.values() if n.node_type == NodeType.ALIAS]
        alias_owner = {e.src: e.dst for e in graph.edges if e.rel == "alias_of"}

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
                    "mentions": [f"user_{self.rng.randint(0, self.config.n_users - 1)}"],
                    "timestamp": 1000 + i,
                }
            )

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
                }
            )

        profiles: list[dict] = []
        for user in users:
            conns = [e.dst for e in graph.edges if e.src == user.node_id and e.rel == "connected_to"][:5]
            profiles.append(
                {
                    "user_id": user.node_id,
                    "name": user.attrs["name"],
                    "org": user.attrs["org"],
                    "location": user.attrs["location"],
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
                    "location": self.rng.choice(["Remote", "Unknown"]),
                    "connections": [],
                    "work_history": [],
                }
            )
        return PlatformViews(microblog_posts, forum_threads, profiles)

    def generate_tasks(self, graph: CanonicalGraph, views: PlatformViews, count: int = 12) -> list[TaskInstance]:
        tasks = self._seeded_tasks(graph)
        target_count = max(count, len(tasks))

        llm_budget = min(
            max(0, self.config.seeding.llm_generated_task_budget),
            max(0, target_count - len(tasks)),
        )
        if self.config.seeding.llm_generate_remaining_tasks and llm_budget > 0:
            tasks.extend(self._llm_generated_tasks(graph, count=llm_budget, start_idx=len(tasks)))

        if len(tasks) < target_count:
            tasks.extend(self._template_tasks(graph, count=target_count - len(tasks), start_idx=len(tasks)))

        return tasks[:target_count]
