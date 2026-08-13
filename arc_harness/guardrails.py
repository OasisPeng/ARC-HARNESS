"""Guardrail abstractions for frame/action/result validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .actions import Action, Frame
from .environment import EnvironmentResult


class GuardrailDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REWRITE = "rewrite"


@dataclass(frozen=True)
class GuardrailResult:
    decision: GuardrailDecision
    reason: str = ""
    action: Action | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_(cls, reason: str = "") -> "GuardrailResult":
        return cls(GuardrailDecision.PASS, reason=reason)

    @classmethod
    def fail(cls, reason: str) -> "GuardrailResult":
        return cls(GuardrailDecision.FAIL, reason=reason)

    @classmethod
    def rewrite(cls, action: Action, reason: str = "") -> "GuardrailResult":
        return cls(GuardrailDecision.REWRITE, reason=reason, action=action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "action": self.action.to_dict() if self.action else None,
            "metadata": self.metadata,
        }


class ActionGuardrail(Protocol):
    def check_action(self, step: int, frame: Frame, action: Action) -> GuardrailResult:
        ...


class FrameGuardrail(Protocol):
    def check_frame(self, frame: Frame, phase: str) -> GuardrailResult:
        ...


class ResultGuardrail(Protocol):
    def check_result(self, step: int, result: EnvironmentResult) -> GuardrailResult:
        ...


class CoordinateBoundsGuardrail:
    """Reject coordinate actions outside the current frame."""

    def check_action(self, step: int, frame: Frame, action: Action) -> GuardrailResult:
        if action.xy is None:
            return GuardrailResult.pass_()
        x, y = action.xy
        if x < 0 or y < 0 or x >= frame.width or y >= frame.height:
            return GuardrailResult.fail(f"Action coordinate {(x, y)} outside frame {frame.width}x{frame.height}.")
        return GuardrailResult.pass_()


class MaxChangedCellsGuardrail:
    """Reject environment results that mutate too many cells at once."""

    def __init__(self, max_changed_cells: int) -> None:
        self.max_changed_cells = max_changed_cells

    def check_result(self, step: int, result: EnvironmentResult) -> GuardrailResult:
        changed = result.info.get("changed_cells")
        if changed is not None and changed > self.max_changed_cells:
            return GuardrailResult.fail(f"Result changed {changed} cells, budget is {self.max_changed_cells}.")
        return GuardrailResult.pass_()

