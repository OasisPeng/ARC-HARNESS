"""Policy and permission helpers for action execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from arc_harness.core.actions import Action


class Decision(str, Enum):
    ALLOW = "allow"
    REWRITE = "rewrite"
    BLOCK = "block"
    ASK = "ask"


@dataclass(frozen=True)
class HookDecision:
    """Decision returned by hooks before an action is executed."""

    decision: Decision
    action: Action | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, action: Action | None = None, reason: str = "") -> "HookDecision":
        return cls(Decision.ALLOW, action=action, reason=reason)

    @classmethod
    def rewrite(cls, action: Action, reason: str = "") -> "HookDecision":
        return cls(Decision.REWRITE, action=action, reason=reason)

    @classmethod
    def block(cls, reason: str = "") -> "HookDecision":
        return cls(Decision.BLOCK, reason=reason)

    @classmethod
    def ask(cls, reason: str = "") -> "HookDecision":
        return cls(Decision.ASK, reason=reason)
