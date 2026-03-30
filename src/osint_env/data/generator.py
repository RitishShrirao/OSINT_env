from __future__ import annotations

import random
from dataclasses import dataclass

from osint_env.domain.models import CanonicalGraph, Edge, EnvironmentConfig, Node, NodeType, TaskInstance


@dataclass(slots=True)
class PlatformViews:
    microblog_posts: list[dict]
    forum_threads: list[dict]
    profiles: list[dict]


class DatasetGenerator:
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng = random.Random(config.seed)

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
                org_name = graph.nodes[edge.dst].attrs["name"]
                q = f"Which user works at {org_name}?"
                a = edge.src
                support = [edge]
            tasks.append(TaskInstance(task_id=f"task_{i}", task_type=mode, question=q, answer=a, supporting_edges=support))
        return tasks
