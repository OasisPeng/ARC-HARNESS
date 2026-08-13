"""Agent interfaces and a tiny baseline heuristic agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from .actions import Action, ActionType, Frame
from .delegation import DelegationConfig, DelegationManager, HandoffController, HandoffRule
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


class DelegatingPlannerAgent(ArcAgent):
    """Agent that closes the loop over perception, exploration, and planning subagents."""

    def __init__(self, delegation: DelegationManager | None = None, config: DelegationConfig | None = None) -> None:
        self.delegation = delegation or DelegationManager.with_default_subagents()
        self.config = config or DelegationConfig(max_retries=1)

    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        tried = memory.working.recent_actions(limit=latest_frame.width * latest_frame.height + 8)
        perception, exploration = self.delegation.delegate_many(
            [
                ("perceive", {"frame": latest_frame}),
                ("explore", {"frame": latest_frame, "tried_actions": tried}),
            ],
            memory,
            config=self.config,
        )
        plan = self.delegation.delegate(
            "plan",
            {
                "frame": latest_frame,
                "perception": perception.output,
                "candidate_actions": exploration.output.get("competition_values", []),
                "tried_actions": tried,
            },
            memory,
            budget=8,
            config=self.config,
        )
        steps = plan.output.get("plan", [])
        if steps:
            memory.add_note(f"Delegating planner selected: {steps[0].get('reason', '')}")
            return Action.from_value(steps[0]["action"])
        return HeuristicAgent().choose_action(frames, latest_frame, memory)


class HandoffAgent(ArcAgent):
    """Wrapper that lets specialist agents take over action selection."""

    def __init__(
        self,
        primary: ArcAgent,
        specialists: dict[str, ArcAgent],
        rules: Iterable[HandoffRule],
        *,
        primary_name: str = "primary",
    ) -> None:
        self.primary = primary
        self.specialists = dict(specialists)
        self.controller = HandoffController(primary_name, rules)
        self.primary_name = primary_name

    @property
    def active_agent_name(self) -> str:
        return self.controller.active_name

    def on_episode_start(self, memory: MemoryManager) -> None:
        self.controller.reset()
        self.primary.on_episode_start(memory)
        for agent in self.specialists.values():
            agent.on_episode_start(memory)

    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        state = {
            "step": len(memory.working.steps),
            "status": latest_frame.status,
            "latest_frame": latest_frame,
            "recent_actions": memory.working.recent_actions(),
            "failures": list(memory.working.failures),
            "hypotheses": list(memory.working.hypotheses),
        }
        self.controller.choose_active(state, memory)
        agent = self.specialists.get(self.controller.active_name, self.primary)
        return agent.choose_action(frames, latest_frame, memory)

    def is_done(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> bool:
        active = self.specialists.get(self.controller.active_name, self.primary)
        return active.is_done(frames, latest_frame, memory)

    def on_episode_end(self, memory: MemoryManager) -> None:
        memory.add_fact(
            f"Handoff route ended with active agent {self.controller.active_name}.",
            category="handoff",
            namespace=("delegation", "handoff"),
            tags=("handoff", self.controller.active_name),
            metadata=self.controller.to_dict(),
        )
        self.primary.on_episode_end(memory)
        for agent in self.specialists.values():
            agent.on_episode_end(memory)
