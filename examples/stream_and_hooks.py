"""Demonstrate streaming events and action rewriting hooks."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_harness import Action, ActionType, ArcThread, EnvironmentResult, Frame, HeuristicAgent, HookDecision


class RewriteFirstClickHook:
    def before_action(self, step: int, frame: Frame, action: Action) -> HookDecision:
        if step == 0:
            return HookDecision.rewrite(Action(ActionType.ACTION6, (2, 1)), "Try known target first in toy game.")
        return HookDecision.allow(action)


class ToyGridGame:
    def __init__(self) -> None:
        self.grid = [[0, 0, 0], [0, 1, 2], [0, 0, 0]]

    def reset(self) -> Frame:
        self.grid = [[0, 0, 0], [0, 1, 2], [0, 0, 0]]
        return Frame.from_grid(self.grid)

    def step(self, action: Action) -> EnvironmentResult:
        if action.kind == ActionType.ACTION6 and action.xy == (2, 1):
            self.grid[1][2] = 3
            return EnvironmentResult(Frame.from_grid(self.grid, status="WIN"), reward=1.0, done=True)
        return EnvironmentResult(Frame.from_grid(self.grid), reward=0.0, done=False)


def main() -> None:
    thread = ArcThread(memory_dir=".arc_memory", hooks=[RewriteFirstClickHook()])
    for event in thread.run_streamed(ToyGridGame(), HeuristicAgent(), max_steps=8):
        print(event["type"], event["episode_id"])


if __name__ == "__main__":
    main()

