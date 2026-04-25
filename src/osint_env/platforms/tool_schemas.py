from __future__ import annotations

from typing import Any


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def build_lookup_tools() -> list[dict[str, Any]]:
    return [
        _tool_schema(
            "search_posts",
            "Search microblog posts by substring over post text, post id, author id, canonical user id, or referenced entity ids/names.",
            {"query": {"type": "string", "description": "Substring to search for in post text."}},
            ["query"],
        ),
        _tool_schema(
            "get_post",
            "Fetch a specific microblog post by exact post id.",
            {"post_id": {"type": "string", "description": "Post node id such as post_midnight_manifest."}},
            ["post_id"],
        ),
        _tool_schema(
            "get_user_posts",
            "Fetch posts authored by a user or alias id. Alias ids are resolved to the canonical user and vice versa.",
            {"user_id": {"type": "string", "description": "User or alias node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "get_mentions",
            "Fetch posts that mention a given canonical user id.",
            {"user_id": {"type": "string", "description": "Canonical user node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "search_threads",
            "Search forum threads by exact topic name.",
            {"topic": {"type": "string", "description": "Thread topic such as security or ai."}},
            ["topic"],
        ),
        _tool_schema(
            "get_thread",
            "Fetch a specific forum thread by id.",
            {"thread_id": {"type": "string", "description": "Thread node id."}},
            ["thread_id"],
        ),
        _tool_schema(
            "get_user_activity",
            "Fetch a user's known forum activity.",
            {"user_id": {"type": "string", "description": "Canonical user node id."}},
            ["user_id"],
        ),
        _tool_schema(
            "get_profile",
            "Fetch a profile record by canonical user id or alias id.",
            {"user_id": {"type": "string", "description": "Canonical user node id or alias id."}},
            ["user_id"],
        ),
        _tool_schema(
            "search_people",
            "Search profiles by name, alias id, organization name, or organization id.",
            {
                "name": {"type": "string", "description": "Optional name substring.", "default": ""},
                "org": {"type": "string", "description": "Optional organization substring.", "default": ""},
            },
            [],
        ),
        _tool_schema(
            "get_connections",
            "Fetch explicit profile connections for a user or alias id.",
            {"user_id": {"type": "string", "description": "Canonical user node id or alias id."}},
            ["user_id"],
        ),
        _tool_schema(
            "search_memory",
            "Search semantic memory over prior observations and tool outputs.",
            {
                "query": {"type": "string", "description": "Memory retrieval query."},
                "k": {"type": "integer", "description": "Top-k matches.", "default": 5},
            },
            ["query"],
        ),
        _tool_schema(
            "search_shared_context",
            "Search the task-local shared context graph carried with the current question.",
            {
                "query": {"type": "string", "description": "Substring query over shared-context node ids and edge fields."},
                "k": {"type": "integer", "description": "Maximum number of node/edge hits to return.", "default": 5},
            },
            ["query"],
        ),
    ]


def build_action_tools() -> list[dict[str, Any]]:
    return build_lookup_tools() + [
        _tool_schema(
            "add_edge",
            "Add a supported graph edge to the working memory graph.",
            {
                "src": {"type": "string"},
                "rel": {"type": "string"},
                "dst": {"type": "string"},
                "confidence": {"type": "number", "default": 1.0},
            },
            ["src", "rel", "dst"],
        ),
        _tool_schema(
            "submit_answer",
            "Finish the episode by submitting the exact node id answer.",
            {"answer": {"type": "string", "description": "Exact node id answer for the task."}},
            ["answer"],
        ),
    ]
