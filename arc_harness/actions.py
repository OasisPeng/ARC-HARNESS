"""Core data structures for ARC-style interaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, Tuple


class ActionType(str, Enum):
    RESET = "RESET"
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
    ACTION6 = "ACTION6"
    ACTION7 = "ACTION7"


@dataclass(frozen=True)
class Action:
    """An action with an optional coordinate payload."""

    kind: ActionType | str
    xy: Optional[Tuple[int, int]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_competition_value(self) -> Any:
        kind = self.kind.value if isinstance(self.kind, ActionType) else str(self.kind)
        if self.xy is None:
            return kind
        return (kind, self.xy[0], self.xy[1])

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind.value if isinstance(self.kind, ActionType) else str(self.kind)
        return {"kind": kind, "xy": self.xy, "meta": self.meta}

    @classmethod
    def from_value(cls, value: Any) -> "Action":
        if isinstance(value, Action):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Sequence) and len(value) == 3:
            return cls(str(value[0]), (int(value[1]), int(value[2])))
        raise TypeError(f"Cannot convert {value!r} to Action")


@dataclass(frozen=True)
class Frame:
    """A frame observed from an ARC-like environment."""

    grid: Tuple[Tuple[int, ...], ...]
    status: str = "NOT_FINISHED"
    raw: Any = None

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[int]], status: str = "NOT_FINISHED", raw: Any = None) -> "Frame":
        return cls(tuple(tuple(int(cell) for cell in row) for row in grid), status=status, raw=raw)

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def to_dict(self, include_grid: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status, "height": self.height, "width": self.width}
        if include_grid:
            payload["grid"] = self.grid
        return payload


@dataclass
class StepRecord:
    step: int
    before: Frame
    action: Action
    after: Frame
    reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_cells(self) -> int:
        count = 0
        for y, row in enumerate(self.before.grid):
            if y >= len(self.after.grid):
                count += len(row)
                continue
            for x, value in enumerate(row):
                if x >= len(self.after.grid[y]) or self.after.grid[y][x] != value:
                    count += 1
        return count

    def to_dict(self, include_grids: bool = True) -> dict[str, Any]:
        return {
            "step": self.step,
            "before": self.before.to_dict(include_grid=include_grids),
            "action": self.action.to_dict(),
            "after": self.after.to_dict(include_grid=include_grids),
            "reward": self.reward,
            "changed_cells": self.changed_cells,
            "info": self.info,
        }
