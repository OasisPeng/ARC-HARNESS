"""Run the harness against downloaded ARC Prize 2026 public games."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_harness import DelegatingPlannerAgent, HeuristicAgent, OfficialSmokeRunner, RunnerConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-files", default="data/arc-prize-2026-arc-agi-3/environment_files")
    parser.add_argument("--memory-dir", default=".arc_memory/public_smoke")
    parser.add_argument("--agent", choices=("heuristic", "delegating"), default="delegating")
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args()

    agent = DelegatingPlannerAgent() if args.agent == "delegating" else HeuristicAgent()
    report = OfficialSmokeRunner(args.environment_files, memory_dir=args.memory_dir).run(
        agent,
        max_games=args.max_games,
        config=RunnerConfig(max_steps=args.max_steps, abort_on_error=False),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.runnable > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
