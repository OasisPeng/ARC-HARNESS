"""Run a real local Qwen candidate-ranking smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_harness import (  # noqa: E402
    ArcThread,
    CandidateRankingAgent,
    EnvironmentResult,
    Frame,
    QwenLocalRanker,
    RunnerConfig,
    Action,
    ActionType,
)


class TinyEnv:
    def __init__(self) -> None:
        self.grid = [[0, 1], [0, 2]]

    def reset(self) -> Frame:
        return Frame.from_grid(self.grid)

    def step(self, action: Action) -> EnvironmentResult:
        if action.kind == ActionType.ACTION6 and action.xy == (1, 0):
            self.grid[0][1] = 3
            return EnvironmentResult(Frame.from_grid(self.grid, status="WIN"), reward=1.0, done=True)
        return EnvironmentResult(Frame.from_grid(self.grid), reward=0.0, done=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="data/models/qwen2.5-0.5b-instruct")
    parser.add_argument("--memory-dir", default="/tmp/arc_harness_qwen_smoke_memory")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    model = QwenLocalRanker(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        device_map=None,
        torch_dtype=None,
    )
    agent = CandidateRankingAgent(model, inject_context=False)
    thread = ArcThread(memory_dir=args.memory_dir)
    result = thread.run_episode(TinyEnv(), agent, config=RunnerConfig(max_steps=2, abort_on_error=False))
    summary = {
        "status": result.status,
        "steps": result.steps,
        "done": result.done,
        "failures": result.summary.get("failures", []),
        "notes": result.summary.get("notes", []),
        "action_effects": result.summary.get("action_effects", []),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.status == "WIN" and not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
