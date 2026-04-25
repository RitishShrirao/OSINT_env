from __future__ import annotations

try:
    from openenv.env import Env
except ImportError:
    class Env:
        """Minimal fallback used when openenv is not installed locally."""

        def __init__(
            self,
            name: str,
            state_space: str,
            action_space: list[str],
            episode_max_length: int,
        ) -> None:
            self.name = name
            self.state_space = state_space
            self.action_space = action_space
            self.episode_max_length = episode_max_length

