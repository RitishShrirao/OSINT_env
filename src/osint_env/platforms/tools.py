from __future__ import annotations

from collections import defaultdict
from typing import Any

from osint_env.data.generator import PlatformViews


class ToolRegistry:
    def __init__(self, views: PlatformViews):
        self.views = views
        self.alias_lookup = dict(getattr(views, "alias_lookup", {}))
        self._index()

    @staticmethod
    def _normalize_lookup_token(value: str) -> str:
        token = str(value or "").strip().lower()
        for prefix in ("org_", "loc_", "event_", "post_", "thr_", "thread_", "alias_", "user_"):
            if token.startswith(prefix):
                token = token[len(prefix) :]
                break
        return token.replace("_", " ")

    def _resolve_user_ids(self, user_id: str) -> list[str]:
        user_id = str(user_id or "").strip()
        if not user_id:
            return []
        resolved = [user_id]
        canonical = self.alias_lookup.get(user_id)
        if canonical and canonical not in resolved:
            resolved.append(canonical)
        for alias_id, owner in self.alias_lookup.items():
            if owner == user_id and alias_id not in resolved:
                resolved.append(alias_id)
        return resolved

    def _index(self) -> None:
        self.posts_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.mentions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.posts_by_id = {post["post_id"]: post for post in self.views.microblog_posts}
        for post in self.views.microblog_posts:
            self.posts_by_user[post["user_id"]].append(post)
            canonical_user = post.get("canonical_user")
            if canonical_user:
                self.posts_by_user[canonical_user].append(post)
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
        needle = str(query or "").lower()
        results = [
            p
            for p in self.views.microblog_posts
            if start <= p["timestamp"] <= end
            and (
                needle in p["text"].lower()
                or needle in str(p.get("post_id", "")).lower()
                or needle in str(p.get("user_id", "")).lower()
                or needle in str(p.get("canonical_user", "")).lower()
                or any(needle in str(ref).lower() for ref in p.get("references", []))
                or any(needle in str(ref).lower() for ref in p.get("reference_names", []))
            )
        ]
        return {"results": results[:20], "count": len(results)}

    def get_post(self, post_id: str) -> dict[str, Any]:
        post = self.posts_by_id.get(post_id)
        return {"result": post, "found": post is not None}

    def get_user_posts(self, user_id: str) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        seen_post_ids: set[str] = set()
        for resolved_id in self._resolve_user_ids(user_id):
            for post in self.posts_by_user.get(resolved_id, []):
                post_id = str(post.get("post_id", ""))
                if post_id in seen_post_ids:
                    continue
                seen_post_ids.add(post_id)
                results.append(post)
        return {"results": results, "count": len(results)}

    def get_mentions(self, user_id: str) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        seen_post_ids: set[str] = set()
        for resolved_id in self._resolve_user_ids(user_id):
            for post in self.mentions_by_user.get(resolved_id, []):
                post_id = str(post.get("post_id", ""))
                if post_id in seen_post_ids:
                    continue
                seen_post_ids.add(post_id)
                results.append(post)
        return {"results": results, "count": len(results)}

    def search_threads(self, topic: str) -> dict[str, Any]:
        needle = str(topic or "").strip().lower()
        results = [
            t
            for t in self.views.forum_threads
            if t["topic"] == topic
            or needle in str(t.get("thread_id", "")).lower()
            or needle in str(t.get("title", "")).lower()
        ]
        return {"results": results[:20], "count": len(results)}

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        thread = self.threads_by_id.get(thread_id)
        return {"result": thread, "found": thread is not None}

    def get_user_activity(self, user_id: str) -> dict[str, Any]:
        acts: list[dict[str, Any]] = []
        seen = set()
        for resolved_id in self._resolve_user_ids(user_id):
            for activity in self.activity_by_user.get(resolved_id, []):
                key = (activity.get("kind"), activity.get("thread_id"))
                if key in seen:
                    continue
                seen.add(key)
                acts.append(activity)
        return {"results": acts, "count": len(acts)}

    def get_profile(self, user_id: str) -> dict[str, Any]:
        resolved_ids = self._resolve_user_ids(user_id)
        profile = next((self.profiles_by_user.get(candidate) for candidate in resolved_ids if self.profiles_by_user.get(candidate)), None)
        return {"result": profile, "found": profile is not None}

    def search_people(self, name: str | None = None, org: str | None = None) -> dict[str, Any]:
        results = self.views.profiles
        if name:
            name_query = str(name).lower()
            results = [
                p
                for p in results
                if name_query in p["name"].lower()
                or name_query in p["user_id"].lower()
                or any(name_query in alias.lower() for alias in p.get("alias_ids", []))
            ]
        if org:
            org_query = str(org).lower()
            normalized_org = self._normalize_lookup_token(org_query)
            results = [
                p
                for p in results
                if org_query in p["org"].lower()
                or org_query in str(p.get("org_id", "")).lower()
                or (normalized_org and normalized_org in p["org"].lower())
            ]
        return {"results": results[:20], "count": len(results)}

    def get_connections(self, user_id: str) -> dict[str, Any]:
        resolved_ids = self._resolve_user_ids(user_id)
        profile = next((self.profiles_by_user.get(candidate) for candidate in resolved_ids if self.profiles_by_user.get(candidate)), None)
        return {"results": profile["connections"] if profile else [], "count": len(profile["connections"]) if profile else 0}
