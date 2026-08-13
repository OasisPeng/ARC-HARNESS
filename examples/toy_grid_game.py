"""Run a tiny ARC-like game through the harness."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_harness import Action, ActionType, ArcThread, EnvironmentResult, Frame, HeuristicAgent, JsonlTraceHook


class ToyGridGame:
    """A 3x3 game where clicking the target at (2, 1) wins."""

    def __init__(self) -> None:
        self.grid = [[0, 0, 0], [0, 1, 2], [0, 0, 0]]
        self.done = False

    def reset(self) -> Frame:
        self.done = False
        self.grid = [[0, 0, 0], [0, 1, 2], [0, 0, 0]]
        return Frame.from_grid(self.grid)

    def step(self, action: Action) -> EnvironmentResult:
        if action.kind == ActionType.ACTION6 and action.xy == (2, 1):
            self.grid[1][2] = 3
            self.done = True
            return EnvironmentResult(Frame.from_grid(self.grid, status="WIN"), reward=1.0, done=True)
        return EnvironmentResult(Frame.from_grid(self.grid), reward=0.0, done=False)


def main() -> None:
    thread = ArcThread(memory_dir=".arc_memory", hooks=[JsonlTraceHook(".arc_memory/traces/toy.jsonl")])
    result = thread.run_episode(ToyGridGame(), HeuristicAgent(), max_steps=16)
    print(f"thread={thread.thread_id} status={result.status} steps={result.steps} done={result.done}")


if __name__ == "__main__":
    main()

