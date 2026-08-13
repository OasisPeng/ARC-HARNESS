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
    info: dict[str, Any] = field(default_factory=dict)


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
                    info=record.get("info", {}),
                )
            )
        return cls(episode_id=episode_id, summary=records[0], steps=steps)

