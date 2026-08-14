"""Error context and recovery data structures."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

from arc_harness.core.actions import Action, Frame


class HarnessError(Exception):
    """Base exception raised by the harness."""


class ValidationError(HarnessError):
    """Raised when frames, actions, or environments violate the harness protocol."""


@dataclass(frozen=True)
class ErrorContext:
    episode_id: str
    step: int | None
    phase: str
    error_type: str
    error: str
    traceback: str
    latest_frame: dict[str, Any] | None = None
    proposed_action: dict[str, Any] | None = None
    recent_actions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_exception(
        cls,
        episode_id: str,
        exc: BaseException,
        *,
        step: int | None = None,
        phase: str = "unknown",
        latest_frame: Frame | None = None,
        proposed_action: Action | None = None,
        recent_actions: list[Action] | None = None,
    ) -> "ErrorContext":
        return cls(
            episode_id=episode_id,
            step=step,
            phase=phase,
            error_type=type(exc).__name__,
            error=repr(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            latest_frame=latest_frame.to_dict(include_grid=False) if latest_frame else None,
            proposed_action=proposed_action.to_dict() if proposed_action else None,
            recent_actions=[action.to_dict() for action in (recent_actions or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "phase": self.phase,
            "error_type": self.error_type,
            "error": self.error,
            "traceback": self.traceback,
            "latest_frame": self.latest_frame,
            "proposed_action": self.proposed_action,
            "recent_actions": self.recent_actions,
        }

