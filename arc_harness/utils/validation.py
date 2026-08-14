"""Validation helpers for ARC frames and actions."""

from __future__ import annotations

from arc_harness.core.actions import Action, ActionType, Frame
from arc_harness.utils.errors import ValidationError


def validate_frame(frame: Frame) -> None:
    if not isinstance(frame, Frame):
        raise ValidationError(f"Expected Frame, got {type(frame).__name__}.")
    if frame.height == 0 or frame.width == 0:
        raise ValidationError("Frame grid must be non-empty.")
    width = frame.width
    for row_index, row in enumerate(frame.grid):
        if len(row) != width:
            raise ValidationError(f"Frame grid row {row_index} has width {len(row)}, expected {width}.")
        for value in row:
            if not isinstance(value, int):
                raise ValidationError(f"Frame cell values must be int, got {type(value).__name__}.")
            if value < 0 or value > 15:
                raise ValidationError(f"Frame cell value {value} is outside ARC color range 0..15.")


def validate_action(action: Action, frame: Frame | None = None) -> None:
    if not isinstance(action, Action):
        raise ValidationError(f"Expected Action, got {type(action).__name__}.")
    valid_kinds = {item.value for item in ActionType}
    kind = action.kind.value if isinstance(action.kind, ActionType) else str(action.kind)
    if kind not in valid_kinds:
        raise ValidationError(f"Unknown action kind: {kind}.")
    if action.xy is not None:
        x, y = action.xy
        if x < 0 or y < 0:
            raise ValidationError(f"Action coordinate must be non-negative, got {(x, y)}.")
        if frame is not None and (x >= frame.width or y >= frame.height):
            raise ValidationError(f"Action coordinate {(x, y)} is outside frame {frame.width}x{frame.height}.")

