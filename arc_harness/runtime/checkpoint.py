"""Lightweight checkpoint persistence for episode recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_harness.core.actions import Frame, StepRecord
from arc_harness.utils.events import utc_now


@dataclass
class CheckpointStore:
    root: Path
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def write(self, episode_id: str, step: int, frame: Frame, records: list[StepRecord], extra: dict[str, Any] | None = None) -> Path | None:
        if not self.enabled:
            return None
        path = self.root / f"{episode_id}.json"
        payload = {
            "episode_id": episode_id,
            "step": step,
            "created_at": utc_now(),
            "frame": frame.to_dict(include_grid=True),
            "recent_records": [record.to_dict(include_grids=False) for record in records[-10:]],
            "extra": extra or {},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read(self, episode_id: str) -> dict[str, Any]:
        return json.loads((self.root / f"{episode_id}.json").read_text(encoding="utf-8"))

    def latest(self) -> dict[str, Any] | None:
        if not self.root.exists():
            return None
        paths = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        if not paths:
            return None
        return json.loads(paths[-1].read_text(encoding="utf-8"))

