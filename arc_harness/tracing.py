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

    def timeline(self, trace_id: str) -> "TraceTimeline":
        return TraceTimeline.from_trace(self.read(trace_id))


@dataclass(frozen=True)
class TraceTimelineItem:
    name: str
    span_id: str
    parent_id: str | None
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TraceTimeline:
    trace_id: str
    workflow_name: str
    group_id: str | None
    status: str
    total_duration_ms: float
    items: list[TraceTimelineItem]

    @classmethod
    def from_trace(cls, trace: dict[str, Any]) -> "TraceTimeline":
        spans = trace.get("spans", [])
        items = [
            TraceTimelineItem(
                name=str(span.get("name", "")),
                span_id=str(span.get("span_id", "")),
                parent_id=span.get("parent_id"),
                duration_ms=float(span.get("duration_ms") or 0.0),
                metadata=dict(span.get("metadata", {})),
            )
            for span in spans
        ]
        return cls(
            trace_id=str(trace.get("trace_id", "")),
            workflow_name=str(trace.get("workflow_name", "")),
            group_id=trace.get("group_id"),
            status=str(trace.get("metadata", {}).get("status", "UNKNOWN")),
            total_duration_ms=sum(item.duration_ms for item in items),
            items=items,
        )

    def stage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.name] = counts.get(item.name, 0) + 1
        return counts

    def slowest(self, limit: int = 5) -> list[TraceTimelineItem]:
        return sorted(self.items, key=lambda item: item.duration_ms, reverse=True)[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "workflow_name": self.workflow_name,
            "group_id": self.group_id,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "stage_counts": self.stage_counts(),
            "items": [item.to_dict() for item in self.items],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Trace {self.trace_id}",
            "",
            f"- workflow: {self.workflow_name}",
            f"- group_id: {self.group_id or ''}",
            f"- status: {self.status}",
            f"- span_count: {len(self.items)}",
            f"- total_duration_ms: {self.total_duration_ms:.3f}",
            f"- stage_counts: {self.stage_counts()}",
            "",
            "| span | duration_ms | parent | metadata |",
            "|---|---:|---|---|",
        ]
        for item in self.items:
            metadata = _compact_metadata(item.metadata)
            lines.append(f"| {item.name} | {item.duration_ms:.3f} | {item.parent_id or ''} | {metadata} |")
        return "\n".join(lines) + "\n"

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def write_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target


def _compact_metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    compact = {}
    for key in ("step", "status", "done", "source", "stage", "action", "changed_cells", "reward", "error_type"):
        if key in metadata:
            compact[key] = metadata[key]
    if not compact:
        compact = {key: metadata[key] for key in sorted(metadata)[:4]}
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)
