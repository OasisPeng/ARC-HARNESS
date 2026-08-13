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


@dataclass
class PlannerSubAgent:
    """Rank candidate actions using perception, prior effects, and exploration state."""

    name: str = "PlannerSubAgent"
    kinds: tuple[str, ...] = ("plan",)

    def run(self, task: SubTask, memory: MemoryManager) -> SubAgentResult:
        frame = _coerce_frame(task.payload.get("frame"))
        tried = set(_action_key(action) for action in _coerce_actions(task.payload.get("tried_actions", ())))
        candidates = _coerce_actions(task.payload.get("candidate_actions", ()))
        if not candidates:
            candidates = _default_candidates(frame, tried)
        perception = task.payload.get("perception", {})
        components = perception.get("components", []) if isinstance(perception, dict) else []
        target_cells = _target_cells_from_components(components)
        scored = []
        prior_effects = memory.working.action_effects()
        for action in candidates:
            if _action_key(action) in tried:
                continue
            score, reason = _score_action(action, target_cells, prior_effects)
            scored.append((score, action, reason))
        scored.sort(key=lambda item: item[0], reverse=True)
        plan = [
            {"action": action.to_dict(), "score": round(score, 3), "reason": reason}
            for score, action, reason in scored[: max(0, task.budget)]
        ]
        output = {
            "plan": plan,
            "candidate_count": len(candidates),
            "planned_count": len(plan),
            "stop_reason": "plan_ready" if plan else "need_more_exploration",
        }
        summary = f"ranked {len(plan)} actions from {len(candidates)} candidates"
        confidence = 0.75 if plan else 0.35
        return SubAgentResult(task.task_id, self.name, True, output, summary, confidence=confidence)


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


def _default_candidates(frame: Frame, tried: set[tuple[str, tuple[int, int] | None]]) -> list[Action]:
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
    return candidates


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


def _target_cells_from_components(components: list[dict[str, Any]]) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for component in components:
        if component.get("color") == 0:
            continue
        bbox = component.get("bbox") or {}
        if {"x1", "y1", "x2", "y2"}.issubset(bbox):
            targets.add(((int(bbox["x1"]) + int(bbox["x2"])) // 2, (int(bbox["y1"]) + int(bbox["y2"])) // 2))
        for cell in component.get("sample_cells", []):
            if "x" in cell and "y" in cell:
                targets.add((int(cell["x"]), int(cell["y"])))
    return targets


def _score_action(action: Action, target_cells: set[tuple[int, int]], prior_effects: list[dict[str, Any]]) -> tuple[float, str]:
    score = 0.2
    reasons = ["untried"]
    kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
    if kind == ActionType.ACTION6.value and action.xy is not None:
        score += 0.25
        reasons.append("coordinate probe")
        if action.xy in target_cells:
            score += 0.35
            reasons.append("touches perceived object")
    for effect in reversed(prior_effects[-8:]):
        action_info = effect.get("action", {})
        if action_info.get("kind") == kind and effect.get("changed_cells", 0) > 0:
            score += 0.15
            reasons.append("same action kind previously changed cells")
            break
        if action_info.get("kind") == kind and effect.get("reward", 0.0) > 0:
            score += 0.25
            reasons.append("same action kind previously earned reward")
            break
    return score, "; ".join(reasons)


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
