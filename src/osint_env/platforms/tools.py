from __future__ import annotations

from collections import defaultdict
from typing import Any

from osint_env.data.generator import PlatformViews


class ToolRegistry:
    def __init__(self, views: PlatformViews):
        self.views = views
        self._index()

    def _index(self) -> None:
        self.posts_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.mentions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for post in self.views.microblog_posts:
            self.posts_by_user[post["user_id"]].append(post)
            for m in post.get("mentions", []):
                self.mentions_by_user[m].append(post)

        self.threads_by_id = {t["thread_id"]: t for t in self.views.forum_threads}
        self.activity_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for thread in self.views.forum_threads:
            self.activity_by_user[thread["author_id"]].append({"kind": "thread", "thread_id": thread["thread_id"]})
            for c in thread.get("comments", []):
                self.activity_by_user[c["user_id"]].append({"kind": "comment", "thread_id": thread["thread_id"]})

        self.profiles_by_user = {p["user_id"]: p for p in self.views.profiles}

    def call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, tool_name, None)
        if not fn:
            raise ValueError(f"Unknown tool: {tool_name}")
        return fn(**args)

    def search_posts(self, query: str, time_range: tuple[int, int] | None = None) -> dict[str, Any]:
        start, end = time_range or (0, 10**9)
        results = [
            p for p in self.views.microblog_posts if query.lower() in p["text"].lower() and start <= p["timestamp"] <= end
        ]
        return {"results": results[:20], "count": len(results)}

    def get_user_posts(self, user_id: str) -> dict[str, Any]:
        return {"results": self.posts_by_user.get(user_id, []), "count": len(self.posts_by_user.get(user_id, []))}

    def get_mentions(self, user_id: str) -> dict[str, Any]:
        return {"results": self.mentions_by_user.get(user_id, []), "count": len(self.mentions_by_user.get(user_id, []))}

    def search_threads(self, topic: str) -> dict[str, Any]:
        results = [t for t in self.views.forum_threads if t["topic"] == topic]
        return {"results": results[:20], "count": len(results)}

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        thread = self.threads_by_id.get(thread_id)
        return {"result": thread, "found": thread is not None}

    def get_user_activity(self, user_id: str) -> dict[str, Any]:
        acts = self.activity_by_user.get(user_id, [])
        return {"results": acts, "count": len(acts)}

    def get_profile(self, user_id: str) -> dict[str, Any]:
        profile = self.profiles_by_user.get(user_id)
        return {"result": profile, "found": profile is not None}

    def search_people(self, name: str | None = None, org: str | None = None) -> dict[str, Any]:
        results = self.views.profiles
        if name:
            results = [p for p in results if name.lower() in p["name"].lower()]
        if org:
            results = [p for p in results if org.lower() in p["org"].lower()]
        return {"results": results[:20], "count": len(results)}

    def get_connections(self, user_id: str) -> dict[str, Any]:
        profile = self.profiles_by_user.get(user_id)
        return {"results": profile["connections"] if profile else [], "count": len(profile["connections"]) if profile else 0}
