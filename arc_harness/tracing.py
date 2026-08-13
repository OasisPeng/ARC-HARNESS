"""Lightweight tracing inspired by OpenAI Agents SDK spans."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import utc_now


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: _id("span"))
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    duration_ms: float | None = None

    def finish(self, metadata: dict[str, Any] | None = None) -> "Span":
        if metadata:
            self.metadata.update(metadata)
        if self.ended_at is None:
            self.ended_at = utc_now()
            self.duration_ms = (time.perf_counter() - self._started_perf) * 1000.0
        return self

    def __post_init__(self) -> None:
        self._started_perf = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class Trace:
    workflow_name: str
    trace_id: str = field(default_factory=lambda: _id("trace"))
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    spans: list[Span] = field(default_factory=list)

    def start_span(self, name: str, parent_id: str | None = None, metadata: dict[str, Any] | None = None) -> Span:
        span = Span(name=name, trace_id=self.trace_id, parent_id=parent_id, metadata=dict(metadata or {}))
        self.spans.append(span)
        return span

    def finish(self, metadata: dict[str, Any] | None = None) -> "Trace":
        if metadata:
            self.metadata.update(metadata)
        if self.ended_at is None:
            self.ended_at = utc_now()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "trace_id": self.trace_id,
            "group_id": self.group_id,
            "metadata": self.metadata,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass
class TraceStore:
    root: Path
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def write(self, trace: Trace) -> Path | None:
        if not self.enabled:
            return None
        path = self.root / f"{trace.trace_id}.json"
        path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def append_jsonl(self, trace: Trace) -> Path | None:
        if not self.enabled:
            return None
        path = self.root / "traces.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return path

    def read(self, trace_id: str) -> dict[str, Any]:
        return json.loads((self.root / f"{trace_id}.json").read_text(encoding="utf-8"))

