"""Recovery policies for stage-based ARC agent loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from arc_harness.core.actions import Action, ActionType


class RecoveryKind(str, Enum):
    RAISE = "raise"
    RETRY = "retry"
    REPLAN = "replan"
    FALLBACK_ACTION = "fallback_action"
    ABORT = "abort"


@dataclass(frozen=True)
class RecoveryDecision:
    """Decision returned when a loop stage fails."""

    kind: RecoveryKind
    reason: str = ""
    action: Action | None = None
    status: str = "RECOVERY_ABORTED"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def raise_error(cls, reason: str = "") -> "RecoveryDecision":
        return cls(RecoveryKind.RAISE, reason=reason)

    @classmethod
    def retry(cls, reason: str = "") -> "RecoveryDecision":
        return cls(RecoveryKind.RETRY, reason=reason)

    @classmethod
    def replan(cls, reason: str = "") -> "RecoveryDecision":
        return cls(RecoveryKind.REPLAN, reason=reason)

    @classmethod
    def fallback_action(cls, action: Action, reason: str = "") -> "RecoveryDecision":
        return cls(RecoveryKind.FALLBACK_ACTION, reason=reason, action=action)

    @classmethod
    def abort(cls, status: str = "RECOVERY_ABORTED", reason: str = "") -> "RecoveryDecision":
        return cls(RecoveryKind.ABORT, reason=reason, status=status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "action": self.action.to_dict() if self.action else None,
            "status": self.status,
            "metadata": self.metadata,
        }


class RecoveryPolicy(Protocol):
    """Protocol for handling failed loop stages."""

    def decide(self, *, error: BaseException, state: Any, runtime: Any, stage: Any, attempt: int) -> RecoveryDecision:
        ...


class NoRecoveryPolicy:
    """Always re-raise stage errors."""

    def decide(self, *, error: BaseException, state: Any, runtime: Any, stage: Any, attempt: int) -> RecoveryDecision:
        return RecoveryDecision.raise_error(repr(error))


@dataclass
class DefaultRecoveryPolicy:
    """Small deterministic recovery policy for offline ARC experiments.

    The default policy retries delegated/planning stages once when configured,
    uses an untried heuristic action when action selection or validation fails,
    and otherwise lets the runner's existing error handling capture the failure.
    """

    max_retries: int = 1
    fallback_on_decision_error: bool = True
    fallback_on_permission_error: bool = True
    abort_on_exhausted_retries: bool = False

    def decide(self, *, error: BaseException, state: Any, runtime: Any, stage: Any, attempt: int) -> RecoveryDecision:
        stage_name = getattr(stage, "name", "")
        if "guardrail failed" in str(error):
            return RecoveryDecision.raise_error(repr(error))
        if stage_name in {"perception", "exploration", "planning"} and attempt <= self.max_retries:
            return RecoveryDecision.retry(f"retry {stage_name} after {type(error).__name__}")
        if stage_name == "planning" and attempt > self.max_retries:
            return RecoveryDecision.fallback_action(_fallback_action(state, runtime), f"fallback after planning {type(error).__name__}")
        if stage_name == "decision" and self.fallback_on_decision_error:
            return RecoveryDecision.fallback_action(_fallback_action(state, runtime), f"fallback after {type(error).__name__}")
        if stage_name == "permission" and self.fallback_on_permission_error:
            return RecoveryDecision.fallback_action(_fallback_action(state, runtime), f"fallback after permission error")
        if self.abort_on_exhausted_retries:
            return RecoveryDecision.abort("RECOVERY_ABORTED", f"abort after {type(error).__name__}")
        return RecoveryDecision.raise_error(repr(error))


def _fallback_action(state: Any, runtime: Any) -> Action:
    frame = state.frame
    tried = {_action_key(action) for action in runtime.memory.working.recent_actions(limit=frame.width * frame.height + 8)}
    for y in range(frame.height):
        for x in range(frame.width):
            candidate = Action(ActionType.ACTION6, (x, y), {"source": "recovery"})
            if _action_key(candidate) not in tried:
                return candidate
    for kind in (ActionType.ACTION1, ActionType.ACTION2, ActionType.ACTION3, ActionType.ACTION4, ActionType.ACTION5):
        candidate = Action(kind, meta={"source": "recovery"})
        if _action_key(candidate) not in tried:
            return candidate
    return Action(ActionType.ACTION1, meta={"source": "recovery"})


def _action_key(action: Action) -> tuple[str, tuple[int, int] | None]:
    kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
    return kind, action.xy
