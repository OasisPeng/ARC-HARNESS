"""Adapters for the official ARC-AGI-3 toolkit.

The official Kaggle data bundle includes the `arc-agi` toolkit and public
`environment_files`. This module keeps that dependency optional: tests and
offline notebooks can still use the harness without importing the official
package, while real submissions can wrap `arc_agi.Arcade().make(...)`.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .actions import Action, ActionType, Frame
from .environment import EnvironmentResult


@dataclass(frozen=True)
class ArcAgi3Config:
    """Configuration for creating an official ARC-AGI-3 environment."""

    game_id: str
    operation_mode: str = "OFFLINE"
    environments_dir: str | Path = "environment_files"
    render_mode: str | None = None
    arc_api_key: str = ""
    arc_base_url: str = "https://three.arcprize.org"
    recordings_dir: str | Path = "recordings"
    metadata: dict[str, Any] = field(default_factory=dict)


class OfficialDependencyError(ImportError):
    """Raised when the optional official ARC-AGI-3 package is unavailable."""


class OfficialArcEnvironment:
    """Harness-compatible wrapper around an official ARC-AGI-3 environment."""

    def __init__(self, env: Any, *, game_id: str | None = None) -> None:
        self.env = env
        self.game_id = game_id or _get_nested(env, "info.game_id") or getattr(env, "game_id", None)
        self.latest_raw = None

    @classmethod
    def from_config(cls, config: ArcAgi3Config) -> "OfficialArcEnvironment":
        arcade = make_arcade(config)
        kwargs = {}
        if config.render_mode is not None:
            kwargs["render_mode"] = config.render_mode
        env = arcade.make(config.game_id, **kwargs)
        if env is None:
            raise RuntimeError(f"Official ARC-AGI-3 toolkit could not create game {config.game_id!r}.")
        return cls(env, game_id=config.game_id)

    @property
    def action_space(self) -> list[str]:
        return [_action_name(action) for action in getattr(self.env, "action_space", [])]

    def reset(self) -> Frame:
        raw = self.env.reset()
        if raw is None:
            raise RuntimeError("Official ARC-AGI-3 env.reset() returned None.")
        self.latest_raw = raw
        return coerce_official_frame(raw)

    def step(self, action: Action) -> EnvironmentResult:
        official_action = resolve_official_action(action, getattr(self.env, "action_space", ()))
        data = {"x": action.xy[0], "y": action.xy[1]} if action.xy is not None else None
        reasoning = {"harness_action": action.to_dict()}
        raw = self.env.step(official_action, data=data, reasoning=reasoning)
        if raw is None:
            raise RuntimeError(f"Official ARC-AGI-3 env.step({action.to_competition_value()!r}) returned None.")
        self.latest_raw = raw
        frame = coerce_official_frame(raw)
        done = frame.status in {"WIN", "GAME_OVER", "DONE"}
        reward = 1.0 if frame.status == "WIN" else 0.0
        return EnvironmentResult(
            frame,
            reward=reward,
            done=done,
            info={
                "game_id": self.game_id,
                "official_action": _action_name(official_action),
                "available_actions": self.action_space,
                "raw_status": _raw_status(raw),
            },
        )


class EnvironmentFileCatalog:
    """Index public `environment_files` folders from the Kaggle bundle."""

    def __init__(self, root: str | Path = "environment_files") -> None:
        self.root = Path(root)

    def list_games(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        games = []
        for metadata_path in sorted(self.root.rglob("metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
            game_id = str(metadata.get("game_id") or metadata.get("id") or metadata_path.parent.name)
            games.append({"game_id": game_id, "path": str(metadata_path.parent), "metadata": metadata})
        return games


def make_arcade(config: ArcAgi3Config) -> Any:
    arc_agi = _optional_import("arc_agi")
    kwargs = {
        "arc_api_key": config.arc_api_key,
        "arc_base_url": config.arc_base_url,
        "environments_dir": str(config.environments_dir),
        "recordings_dir": str(config.recordings_dir),
    }
    operation_mode = _resolve_operation_mode(config.operation_mode)
    if operation_mode is not None:
        kwargs["operation_mode"] = operation_mode
    try:
        return arc_agi.Arcade(**kwargs)
    except TypeError:
        compact = {key: value for key, value in kwargs.items() if value not in {"", None}}
        return arc_agi.Arcade(**compact)


def create_official_environment(config: ArcAgi3Config) -> OfficialArcEnvironment:
    return OfficialArcEnvironment.from_config(config)


def resolve_official_action(action: Action, action_space: Iterable[Any] = ()) -> Any:
    kind = _action_name(action)
    for candidate in action_space:
        if _action_name(candidate) == kind:
            return candidate
    try:
        game_action = _optional_import("arcengine").GameAction
        return getattr(game_action, kind)
    except (OfficialDependencyError, AttributeError):
        return kind


def coerce_official_frame(raw: Any) -> Frame:
    grid = _extract_grid(raw)
    status = _normalize_status(_raw_status(raw))
    return Frame.from_grid(grid, status=status, raw=raw)


def _optional_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise OfficialDependencyError(
            f"Optional official ARC-AGI-3 dependency {module_name!r} is not installed. "
            "Install/use the Kaggle-provided arc_agi_3_wheels bundle or pass an already-created env."
        ) from exc


def _resolve_operation_mode(name: str) -> Any:
    try:
        operation_mode = _optional_import("arc_agi").OperationMode
    except OfficialDependencyError:
        return None
    return getattr(operation_mode, name, getattr(operation_mode, name.upper(), None))


def _extract_grid(raw: Any) -> Any:
    if isinstance(raw, dict):
        for key in ("grid", "frame", "cells", "observation"):
            if key in raw and _looks_like_grid(raw[key]):
                return raw[key]
        state = raw.get("state")
        if _looks_like_grid(state):
            return state
    for attr in ("grid", "frame", "cells", "observation"):
        if hasattr(raw, attr):
            value = getattr(raw, attr)
            if _looks_like_grid(value):
                return value
    if hasattr(raw, "state") and _looks_like_grid(getattr(raw, "state")):
        return getattr(raw, "state")
    if _looks_like_grid(raw):
        return raw
    raise ValueError(f"Could not extract grid from official frame {raw!r}.")


def _raw_status(raw: Any) -> Any:
    if isinstance(raw, dict):
        for key in ("status", "game_state", "state"):
            value = raw.get(key)
            if value is not None and not _looks_like_grid(value):
                return value
    for attr in ("status", "game_state", "state"):
        if hasattr(raw, attr):
            value = getattr(raw, attr)
            if not _looks_like_grid(value):
                return value
    return "NOT_FINISHED"


def _normalize_status(value: Any) -> str:
    if value is None:
        return "NOT_FINISHED"
    if hasattr(value, "name"):
        return str(value.name)
    if hasattr(value, "value") and not isinstance(value, str):
        return _normalize_status(value.value)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def _action_name(value: Any) -> str:
    if isinstance(value, Action):
        kind = value.kind
        return kind.value if hasattr(kind, "value") else str(kind)
    if hasattr(value, "name"):
        return str(value.name)
    if hasattr(value, "value") and not isinstance(value, str):
        return str(value.value)
    return str(value)


def _looks_like_grid(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, dict)):
        return False
    try:
        rows = list(value)
    except TypeError:
        return False
    if not rows:
        return False
    try:
        first = list(rows[0])
    except TypeError:
        return False
    return bool(first) and all(isinstance(cell, (int, float)) for cell in first)


def _get_nested(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current
