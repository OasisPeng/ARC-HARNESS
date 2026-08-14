"""Adapters for Kaggle-style ARC-AGI-3 functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from arc_harness.core.actions import Action, Frame
from arc_harness.core.agent import ArcAgent
from arc_harness.memory.memory import DurableMemory, MemoryManager


class KaggleAgentAdapter:
    """Expose an ArcAgent as choose_action/is_done functions.

    The adapter keeps memory in process. In a Kaggle Notebook, instantiate one
    global adapter and forward the competition functions to it.
    """

    def __init__(self, agent: ArcAgent, memory_dir: str | Path = "/tmp/arc_harness_memory") -> None:
        self.agent = agent
        self.memory = MemoryManager(DurableMemory(Path(memory_dir)))

    def is_done(self, frames: Sequence[Any], latest_frame: Any) -> bool:
        parsed_frames = [coerce_frame(frame) for frame in frames]
        parsed_latest = coerce_frame(latest_frame)
        return self.agent.is_done(parsed_frames, parsed_latest, self.memory)

    def choose_action(self, frames: Sequence[Any], latest_frame: Any) -> Any:
        parsed_frames = [coerce_frame(frame) for frame in frames]
        parsed_latest = coerce_frame(latest_frame)
        action = self.agent.choose_action(parsed_frames, parsed_latest, self.memory)
        return action.to_competition_value()


def coerce_frame(value: Any) -> Frame:
    if isinstance(value, Frame):
        return value
    if isinstance(value, dict):
        grid = value.get("grid") or value.get("state") or value.get("frame")
        status = value.get("status", "NOT_FINISHED")
        return Frame.from_grid(grid, status=status, raw=value)
    if hasattr(value, "grid"):
        status = getattr(value, "status", "NOT_FINISHED")
        return Frame.from_grid(getattr(value, "grid"), status=status, raw=value)
    return Frame.from_grid(value, raw=value)

