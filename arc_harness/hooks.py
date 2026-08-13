"""Hook system for observing and controlling the agent loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .actions import Action, Frame, StepRecord
from .events import AgentEvent
from .policy import Decision, HookDecision


class Hook(Protocol):
    def before_episode(self, episode_id: str) -> None:
        ...

    def after_observe(self, frame: Frame) -> None:
        ...

    def before_action(self, step: int, frame: Frame, action: Action) -> Action | HookDecision:
        ...

    def after_action(self, record: StepRecord) -> None:
        ...

    def after_episode(self, episode_id: str, summary: dict) -> None:
        ...

    def on_error(self, episode_id: str, error: BaseException) -> None:
        ...


class HookManager:
    def __init__(self, hooks: Iterable[Hook | "HookMatcher"] | None = None) -> None:
        self.hooks = [hook if isinstance(hook, HookMatcher) else HookMatcher(hook=hook) for hook in (hooks or [])]

    def before_episode(self, episode_id: str) -> None:
        for matcher in self.hooks:
            if not matcher.matches("before_episode"):
                continue
            method = getattr(matcher.hook, "before_episode", None)
            if method:
                method(episode_id)

    def after_observe(self, frame: Frame) -> None:
        for matcher in self.hooks:
            if not matcher.matches("after_observe", status=frame.status):
                continue
            method = getattr(matcher.hook, "after_observe", None)
            if method:
                method(frame)

    def before_action(self, step: int, frame: Frame, action: Action) -> HookDecision:
        next_action = action
        rewritten_reason = ""
        for matcher in self.hooks:
            if not matcher.matches("before_action", action=next_action):
                continue
            method = getattr(matcher.hook, "before_action", None)
            if method:
                result = method(step, frame, next_action)
                decision = _coerce_decision(result, next_action)
                if decision.decision in {Decision.BLOCK, Decision.ASK}:
                    return decision
                if decision.decision == Decision.REWRITE:
                    rewritten_reason = decision.reason
                next_action = decision.action or next_action
        if rewritten_reason:
            return HookDecision.rewrite(next_action, rewritten_reason)
        return HookDecision.allow(next_action)

    def after_action(self, record: StepRecord) -> None:
        for matcher in self.hooks:
            if not matcher.matches("after_action", action=record.action, status=record.after.status):
                continue
            method = getattr(matcher.hook, "after_action", None)
            if method:
                method(record)

    def after_episode(self, episode_id: str, summary: dict) -> None:
        for matcher in self.hooks:
            if not matcher.matches("after_episode", status=str(summary.get("status", ""))):
                continue
            method = getattr(matcher.hook, "after_episode", None)
            if method:
                method(episode_id, summary)

    def on_error(self, episode_id: str, error: BaseException) -> None:
        for matcher in self.hooks:
            if not matcher.matches("on_error"):
                continue
            method = getattr(matcher.hook, "on_error", None)
            if method:
                method(episode_id, error)

    def emit(self, event: AgentEvent) -> None:
        for matcher in self.hooks:
            if not matcher.matches("on_event", event_type=event.type):
                continue
            method = getattr(matcher.hook, "on_event", None)
            if method:
                method(event)


class JsonlTraceHook:
    """Write a compact JSONL trace for debugging and replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def before_episode(self, episode_id: str) -> None:
        self._write({"event": "before_episode", "episode_id": episode_id})

    def after_observe(self, frame: Frame) -> None:
        self._write({"event": "after_observe", "status": frame.status, "size": [frame.width, frame.height]})

    def before_action(self, step: int, frame: Frame, action: Action) -> Action:
        self._write({"event": "before_action", "step": step, "action": action.to_competition_value()})
        return action

    def after_action(self, record: StepRecord) -> None:
        self._write({
            "event": "after_action",
            "step": record.step,
            "reward": record.reward,
            "status": record.after.status,
            "changed_cells": record.changed_cells,
        })

    def after_episode(self, episode_id: str, summary: dict) -> None:
        self._write({"event": "after_episode", "episode_id": episode_id, "summary": summary})

    def on_error(self, episode_id: str, error: BaseException) -> None:
        self._write({"event": "error", "episode_id": episode_id, "error": repr(error)})

    def on_event(self, event: AgentEvent) -> None:
        self._write({"event": "agent_event", **event.to_dict()})


class ActionBudgetHook:
    """Block execution after a fixed number of actions.

    This is useful for debugging planner loops and mirrors the budget controls
    exposed by richer agent SDKs.
    """

    def __init__(self, max_actions: int) -> None:
        self.max_actions = max_actions

    def before_action(self, step: int, frame: Frame, action: Action) -> HookDecision:
        if step >= self.max_actions:
            return HookDecision.block(f"Action budget {self.max_actions} reached.")
        return HookDecision.allow(action)


@dataclass(frozen=True)
class HookMatcher:
    """Filter hooks by event, action kind, frame status, or event type.

    Regex fields mirror Claude-style matchers but stay ARC-specific and
    dependency-free.
    """

    hook: Hook
    event: str | None = None
    action: str | None = None
    status: str | None = None
    event_type: str | None = None
    metadata: dict = field(default_factory=dict)

    def matches(
        self,
        event: str,
        *,
        action: Action | None = None,
        status: str | None = None,
        event_type: str | None = None,
    ) -> bool:
        if self.event and not re.fullmatch(self.event, event):
            return False
        if self.action:
            if action is None:
                return False
            kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
            if not re.fullmatch(self.action, kind):
                return False
        if self.status and not re.fullmatch(self.status, status or ""):
            return False
        if self.event_type and not re.fullmatch(self.event_type, event_type or ""):
            return False
        return True


def _coerce_decision(value: Action | HookDecision | None, default_action: Action) -> HookDecision:
    if isinstance(value, HookDecision):
        return value
    if isinstance(value, Action):
        return HookDecision.allow(value)
    if value is None:
        return HookDecision.allow(default_action)
    raise TypeError(f"Unsupported hook decision: {value!r}")
