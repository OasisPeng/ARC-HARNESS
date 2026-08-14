"""Typed replay helpers for recorded ARC episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayStep:
    step: int
    action: dict[str, Any]
    reward: float
    changed_cells: int
    status: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def action_text(self) -> str:
        kind = self.action.get("kind", "UNKNOWN")
        xy = self.action.get("xy")
        if xy is None:
            return str(kind)
        return f"{kind}({xy[0]},{xy[1]})"

    @property
    def progressed(self) -> bool:
        return self.changed_cells > 0 or self.reward > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "reward": self.reward,
            "changed_cells": self.changed_cells,
            "status": self.status,
            "before": self.before,
            "after": self.after,
            "info": self.info,
        }


@dataclass(frozen=True)
class ReplayEpisode:
    episode_id: str
    summary: dict[str, Any]
    steps: list[ReplayStep]

    @property
    def status(self) -> str:
        return str(self.summary.get("status", "UNKNOWN"))

    @property
    def done(self) -> bool:
        return bool(self.summary.get("done", False))

    @property
    def total_reward(self) -> float:
        return float(self.summary.get("total_reward", 0.0))

    def action_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.action_text] = counts.get(step.action_text, 0) + 1
        return counts

    def progress_steps(self) -> list[ReplayStep]:
        return [step for step in self.steps if step.progressed]

    def noop_steps(self) -> list[ReplayStep]:
        return [step for step in self.steps if not step.progressed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "summary": self.summary,
            "action_counts": self.action_counts(),
            "progress_step_count": len(self.progress_steps()),
            "noop_step_count": len(self.noop_steps()),
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_markdown(self, *, include_grids: bool = False) -> str:
        lines = [
            f"# Replay {self.episode_id}",
            "",
            f"- status: {self.status}",
            f"- done: {self.done}",
            f"- steps: {len(self.steps)}",
            f"- total_reward: {self.total_reward:.3f}",
            f"- trace_id: {self.summary.get('trace_id') or ''}",
            f"- action_counts: {self.action_counts()}",
            "",
            "| step | action | reward | changed_cells | status | progressed |",
            "|---:|---|---:|---:|---|---:|",
        ]
        for step in self.steps:
            lines.append(
                f"| {step.step} | {step.action_text} | {step.reward:.3f} | "
                f"{step.changed_cells} | {step.status} | {step.progressed} |"
            )
            if include_grids:
                lines.extend(["", "before:", "```text", _grid_text(step.before.get("grid")), "```"])
                lines.extend(["after:", "```text", _grid_text(step.after.get("grid")), "```"])
        return "\n".join(lines) + "\n"

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def write_markdown(self, path: str | Path, *, include_grids: bool = False) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(include_grids=include_grids), encoding="utf-8")
        return target

    @classmethod
    def from_jsonl(cls, episode_id: str, path: str | Path) -> "ReplayEpisode":
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records or records[0].get("type") != "summary":
            raise ValueError(f"Episode replay {path} is missing a summary record.")
        steps = []
        for record in records[1:]:
            after = record.get("after", {})
            steps.append(
                ReplayStep(
                    step=int(record["step"]),
                    action=record.get("action", {}),
                    reward=float(record.get("reward", 0.0)),
                    changed_cells=int(record.get("changed_cells", 0)),
                    status=str(after.get("status", "UNKNOWN")),
                    before=record.get("before", {}),
                    after=after,
                    info=record.get("info", {}),
                )
            )
        return cls(episode_id=episode_id, summary=records[0], steps=steps)


def _grid_text(grid: Any) -> str:
    if not grid:
        return ""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)
