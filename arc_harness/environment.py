"""Environment protocol used by the episode runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .actions import Action, Frame


@dataclass
class EnvironmentResult:
    frame: Frame
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


class ArcEnvironment(Protocol):
    """Minimal interface for ARC-like interactive environments."""

    def reset(self) -> Frame:
        ...

    def step(self, action: Action) -> EnvironmentResult:
        ...


def validate_environment(env: ArcEnvironment) -> None:
    """Fail fast when an environment does not match the harness protocol."""

    if not callable(getattr(env, "reset", None)):
        raise TypeError("Environment must define reset() -> Frame.")
    if not callable(getattr(env, "step", None)):
        raise TypeError("Environment must define step(action) -> EnvironmentResult.")
