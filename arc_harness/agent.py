"""Agent interfaces and a tiny baseline heuristic agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from .actions import Action, ActionType, Frame
from .memory import MemoryManager, WorkingMemory


class ArcAgent(ABC):
    """Base class for agents used by the harness."""

    @abstractmethod
    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        ...

    def is_done(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> bool:
        return latest_frame.status in {"WIN", "GAME_OVER", "DONE"}

    def on_episode_start(self, memory: MemoryManager) -> None:
        pass

    def on_episode_end(self, memory: MemoryManager) -> None:
        pass


class HeuristicAgent(ArcAgent):
    """A simple coverage-seeking baseline.

    The agent cycles through available non-coordinate actions, then scans
    coordinate actions across the grid. It is deliberately small, but it gives
    the harness a useful default policy for smoke tests and instrumentation.
    """

    def __init__(self, actions: Iterable[ActionType | str] | None = None) -> None:
        self.actions = list(actions or [
            ActionType.ACTION1,
            ActionType.ACTION2,
            ActionType.ACTION3,
            ActionType.ACTION4,
            ActionType.ACTION5,
        ])
        self._cursor = 0

    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        working: WorkingMemory = memory.working
        if latest_frame.width and latest_frame.height:
            for y in range(latest_frame.height):
                for x in range(latest_frame.width):
                    candidate = Action(ActionType.ACTION6, (x, y))
                    if not working.was_action_tried(candidate):
                        return candidate

        action = Action(self.actions[self._cursor % len(self.actions)])
        self._cursor += 1
        return action


@dataclass(frozen=True)
class FrameDiff:
    changed_cells: int
    status_changed: bool
    new_status: str


class RuleLearningAgent(HeuristicAgent):
    """Baseline that records action effects and avoids exact repeats.

    It is still intentionally simple, but it demonstrates the intended shape of
    future planners: observe effects, update hypotheses, prune repeated actions,
    and prefer actions that previously changed the environment.
    """

    def on_episode_start(self, memory: MemoryManager) -> None:
        memory.add_note("Started rule-learning episode.")

    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        self._learn_from_last_step(memory)
        changing = self._last_changing_action(memory)
        if changing and not memory.working.was_action_tried(changing):
            return changing
        return super().choose_action(frames, latest_frame, memory)

    def _learn_from_last_step(self, memory: MemoryManager) -> None:
        if not memory.working.steps:
            return
        last = memory.working.steps[-1]
        diff = FrameDiff(
            changed_cells=last.changed_cells,
            status_changed=last.before.status != last.after.status,
            new_status=last.after.status,
        )
        if diff.changed_cells > 0:
            memory.add_hypothesis(f"{last.action.to_competition_value()} changed {diff.changed_cells} cells.")
        if diff.status_changed:
            memory.add_hypothesis(f"{last.action.to_competition_value()} changed status to {diff.new_status}.")

    def _last_changing_action(self, memory: MemoryManager) -> Action | None:
        for step in reversed(memory.working.steps):
            if step.changed_cells > 0 and step.after.status not in {"GAME_OVER"}:
                return step.action
        return None
