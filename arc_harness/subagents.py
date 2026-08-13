"""Default deterministic ARC subagents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from .actions import Action, ActionType, Frame
from .delegation import SubAgentResult, SubTask
from .memory import MemoryManager


@dataclass
class PerceptionSubAgent:
    """Summarize visible grid structure: colors and connected components."""

    name: str = "PerceptionSubAgent"
    kinds: tuple[str, ...] = ("perceive",)

    def run(self, task: SubTask, memory: MemoryManager) -> SubAgentResult:
        frame = _coerce_frame(task.payload.get("frame"))
        colors = sorted({cell for row in frame.grid for cell in row})
        components = _connected_components(frame)
        output = {
            "status": frame.status,
            "width": frame.width,
            "height": frame.height,
            "colors": colors,
            "component_count": len(components),
            "components": components[: task.budget],
        }
        summary = (
            f"frame {frame.width}x{frame.height}, status={frame.status}, "
            f"colors={colors}, components={len(components)}"
        )
        return SubAgentResult(task.task_id, self.name, True, output, summary, confidence=0.95)


@dataclass
class DiffSubAgent:
    """Summarize the state transition caused by one action."""

    name: str = "DiffSubAgent"
    kinds: tuple[str, ...] = ("diff",)

    def run(self, task: SubTask, memory: MemoryManager) -> SubAgentResult:
        before = _coerce_frame(task.payload.get("before"))
        after = _coerce_frame(task.payload.get("after"))
        action = _coerce_action(task.payload.get("action"))
        changed = _changed_cells(before, after)
        bbox = _bbox([(item["x"], item["y"]) for item in changed])
        output = {
            "action": action.to_dict() if action else None,
            "changed_count": len(changed),
            "changed_cells": changed[: task.budget],
            "bbox": bbox,
            "status_before": before.status,
            "status_after": after.status,
            "status_changed": before.status != after.status,
        }
        action_text = action.to_competition_value() if action else "unknown action"
        summary = (
            f"{action_text} changed {len(changed)} cells"
            f"{' and status to ' + after.status if before.status != after.status else ''}"
        )
        return SubAgentResult(task.task_id, self.name, True, output, summary, confidence=0.95)


@dataclass
class ExplorerSubAgent:
    """Suggest untried exploration actions for the current frame."""

    name: str = "ExplorerSubAgent"
    kinds: tuple[str, ...] = ("explore",)

    def run(self, task: SubTask, memory: MemoryManager) -> SubAgentResult:
        frame = _coerce_frame(task.payload.get("frame"))
        tried = set(_action_key(action) for action in _coerce_actions(task.payload.get("tried_actions", ())))
        candidates: list[Action] = []
        for y in range(frame.height):
            for x in range(frame.width):
                candidate = Action(ActionType.ACTION6, (x, y))
                if _action_key(candidate) not in tried:
                    candidates.append(candidate)
        for kind in (ActionType.ACTION1, ActionType.ACTION2, ActionType.ACTION3, ActionType.ACTION4, ActionType.ACTION5):
            candidate = Action(kind)
            if _action_key(candidate) not in tried:
                candidates.append(candidate)
        limited = candidates[: max(0, task.budget)]
        output = {
            "candidate_count": len(candidates),
            "candidate_actions": [action.to_dict() for action in limited],
            "competition_values": [action.to_competition_value() for action in limited],
        }
        summary = f"suggested {len(limited)} of {len(candidates)} untried actions"
        return SubAgentResult(task.task_id, self.name, True, output, summary, confidence=0.8)


def _coerce_frame(value: Any) -> Frame:
    if isinstance(value, Frame):
        return value
    if isinstance(value, dict):
        grid = value.get("grid") or value.get("state") or value.get("frame")
        status = value.get("status", "NOT_FINISHED")
        if grid is None:
            raise ValueError("Frame payload must include grid/state/frame.")
        return Frame.from_grid(grid, status=status, raw=value)
    if hasattr(value, "grid"):
        return Frame.from_grid(getattr(value, "grid"), status=getattr(value, "status", "NOT_FINISHED"), raw=value)
    return Frame.from_grid(value)


def _coerce_action(value: Any) -> Action | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return Action(value.get("kind"), tuple(value["xy"]) if value.get("xy") is not None else None, dict(value.get("meta", {})))
    return Action.from_value(value)


def _coerce_actions(values: Iterable[Any]) -> list[Action]:
    return [action for value in values if (action := _coerce_action(value)) is not None]


def _action_key(action: Action) -> tuple[str, tuple[int, int] | None]:
    kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
    return kind, action.xy


def _connected_components(frame: Frame) -> list[dict[str, Any]]:
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y, row in enumerate(frame.grid):
        for x, color in enumerate(row):
            if (x, y) in visited:
                continue
            cells = _flood_fill(frame, x, y, color, visited)
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": _bbox(cells),
                    "sample_cells": [{"x": cx, "y": cy} for cx, cy in cells[:8]],
                }
            )
    components.sort(key=lambda item: (-item["size"], item["color"], item["bbox"]["y1"], item["bbox"]["x1"]))
    return components


def _flood_fill(frame: Frame, start_x: int, start_y: int, color: int, visited: set[tuple[int, int]]) -> list[tuple[int, int]]:
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    visited.add((start_x, start_y))
    cells: list[tuple[int, int]] = []
    while queue:
        x, y = queue.popleft()
        cells.append((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or ny >= frame.height or nx >= frame.width:
                continue
            if (nx, ny) in visited or frame.grid[ny][nx] != color:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))
    return cells


def _changed_cells(before: Frame, after: Frame) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    height = max(before.height, after.height)
    width = max(before.width, after.width)
    for y in range(height):
        for x in range(width):
            before_value = before.grid[y][x] if y < before.height and x < before.width else None
            after_value = after.grid[y][x] if y < after.height and x < after.width else None
            if before_value != after_value:
                changed.append({"x": x, "y": y, "before": before_value, "after": after_value})
    return changed


def _bbox(cells: list[tuple[int, int]]) -> dict[str, int] | None:
    if not cells:
        return None
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
