from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MyEnvV4Action:
    message: str


@dataclass(slots=True)
class _EchoObservation:
    echoed_message: str


@dataclass(slots=True)
class _EchoResult:
    observation: _EchoObservation
    reward: float = 0.0
    done: bool = False


class MyEnvV4Env:
    def __init__(self) -> None:
        self._step_count = 0

    @classmethod
    async def from_docker_image(cls, image_name: str | None = None) -> "MyEnvV4Env":
        return cls()

    async def reset(self) -> _EchoResult:
        self._step_count = 0
        return _EchoResult(observation=_EchoObservation(echoed_message=""), reward=0.0, done=False)

    async def step(self, action: MyEnvV4Action) -> _EchoResult:
        self._step_count += 1
        message = str(getattr(action, "message", ""))
        reward = len(message) * 0.1
        return _EchoResult(
            observation=_EchoObservation(echoed_message=message),
            reward=reward,
            done=False,
        )

    async def close(self) -> None:
        return None
