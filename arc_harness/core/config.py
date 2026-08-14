"""Configuration objects for harness runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunnerConfig:
    """Stable run configuration used by EpisodeRunner and ArcThread.

    Keep this object small and JSON-friendly. More advanced solver settings
    should live in agent/planner-specific configs instead of leaking into the
    harness core.
    """

    max_steps: int = 256
    loop_window: int = 6
    stop_on_loop: bool = False
    validate_frames: bool = True
    validate_actions: bool = True
    checkpoint: bool = True
    abort_on_error: bool = True
    tracing: bool = True
    trace_workflow_name: str = "ARC harness episode"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.loop_window <= 0:
            raise ValueError("loop_window must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "loop_window": self.loop_window,
            "stop_on_loop": self.stop_on_loop,
            "validate_frames": self.validate_frames,
            "validate_actions": self.validate_actions,
            "checkpoint": self.checkpoint,
            "abort_on_error": self.abort_on_error,
            "tracing": self.tracing,
            "trace_workflow_name": self.trace_workflow_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_value(cls, value: "RunnerConfig | int | None") -> "RunnerConfig":
        if value is None:
            return cls()
        if isinstance(value, RunnerConfig):
            return value
        if isinstance(value, int):
            return cls(max_steps=value)
        raise TypeError(f"Cannot build RunnerConfig from {value!r}")
