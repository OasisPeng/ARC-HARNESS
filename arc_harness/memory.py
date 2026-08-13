"""Layered memory inspired by Hermes/pi-hermes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .actions import Action, Frame, StepRecord
from .events import utc_now
from .memory_store import SearchResult, StructuredMemoryStore
from .replay import ReplayEpisode


@dataclass
class WorkingMemory:
    """Short-term per-episode memory."""

    frames: list[Frame] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.frames.clear()
        self.steps.clear()
        self.hypotheses.clear()
        self.failures.clear()
        self.notes.clear()

    def remember_frame(self, frame: Frame) -> None:
        self.frames.append(frame)

    def remember_step(self, record: StepRecord) -> None:
        self.steps.append(record)
        self.frames.append(record.after)

    def was_action_tried(self, action: Action) -> bool:
        return any(step.action == action for step in self.steps)

    def recent_actions(self, limit: int = 8) -> list[Action]:
        return [step.action for step in self.steps[-limit:]]

    def action_effects(self) -> list[dict[str, Any]]:
        return [
            {
                "step": step.step,
                "action": step.action.to_dict(),
                "changed_cells": step.changed_cells,
                "reward": step.reward,
                "status": step.after.status,
            }
            for step in self.steps
        ]

    def detects_loop(self, window: int = 6) -> bool:
        if len(self.frames) < window * 2:
            return False
        return self.frames[-window:] == self.frames[-window * 2:-window]


@dataclass(frozen=True)
class MemoryEntry:
    text: str
    category: str = "fact"
    scope: str = "global"
    tags: tuple[str, ...] = ()
    namespace: tuple[str, ...] = ("global",)
    confidence: float = 1.0
    importance: float = 0.5
    source_episode_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_markdown(self) -> str:
        tag_text = ",".join(self.tags)
        return f"\n§ {self.category} | {self.scope} | {self.created_at} | {tag_text}\n{self.text.strip()}\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "category": self.category,
            "scope": self.scope,
            "tags": list(self.tags),
            "namespace": list(self.namespace),
            "confidence": self.confidence,
            "importance": self.importance,
            "source_episode_id": self.source_episode_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class DurableMemory:
    """Human-readable and replayable durable memory."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "episodes").mkdir(exist_ok=True)
        (self.root / "skills").mkdir(exist_ok=True)
        for name in ("MEMORY.md", "FAILURES.md", "PROCEDURES.md"):
            path = self.root / name
            if not path.exists():
                path.write_text(f"# {name[:-3].title()}\n\n", encoding="utf-8")
        self.store = StructuredMemoryStore(self.root / "memory.db")

    def add_entry(self, entry: MemoryEntry) -> None:
        target = "FAILURES.md" if entry.category == "failure" else "PROCEDURES.md" if entry.category == "procedure" else "MEMORY.md"
        with (self.root / target).open("a", encoding="utf-8") as fh:
            fh.write(entry.to_markdown())
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self.store.put(
            text=entry.text,
            category=entry.category,
            namespace=entry.namespace,
            scope=entry.scope,
            tags=entry.tags,
            metadata=entry.metadata,
            confidence=entry.confidence,
            importance=entry.importance,
            source_episode_id=entry.source_episode_id,
        )

    def add_fact(
        self,
        text: str,
        category: str = "fact",
        scope: str = "global",
        tags: Iterable[str] = (),
        namespace: Iterable[str] = ("global",),
        confidence: float = 1.0,
        importance: float = 0.5,
        source_episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.add_entry(
            MemoryEntry(
                text=text,
                category=category,
                scope=scope,
                tags=tuple(tags),
                namespace=tuple(namespace),
                confidence=confidence,
                importance=importance,
                source_episode_id=source_episode_id,
                metadata=dict(metadata or {}),
            )
        )

    def add_failure(self, text: str, source_episode_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.add_entry(
            MemoryEntry(
                text=text,
                category="failure",
                tags=("failure",),
                namespace=("global", "failures"),
                source_episode_id=source_episode_id,
                metadata=dict(metadata or {}),
                importance=0.8,
            )
        )

    def save_skill(self, name: str, body: str, scope: str = "global", description: str = "") -> Path:
        slug = _slugify(name)
        path = self.root / "skills" / slug
        path.mkdir(parents=True, exist_ok=True)
        text = (
            f"---\nname: {slug}\ndescription: {description}\nscope: {scope}\n---\n\n"
            f"# {name}\n\n{body.strip()}\n"
        )
        skill_path = path / "SKILL.md"
        skill_path.write_text(text, encoding="utf-8")
        self.add_fact(f"Saved procedure skill `{slug}`: {description or name}", category="procedure", scope=scope, tags=("skill", slug))
        return skill_path

    def read_skill(self, name: str) -> str:
        path = self.root / "skills" / _slugify(name) / "SKILL.md"
        return path.read_text(encoding="utf-8")

    def write_episode(self, episode_id: str, records: Iterable[StepRecord], summary: dict) -> Path:
        path = self.root / "episodes" / f"{episode_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "summary", "created_at": utc_now(), **summary}, ensure_ascii=False) + "\n")
            for record in records:
                fh.write(json.dumps(_step_to_json(record), ensure_ascii=False) + "\n")
        return path

    def read_episode(self, episode_id: str) -> list[dict[str, Any]]:
        path = self.root / "episodes" / f"{episode_id}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_replay(self, episode_id: str) -> ReplayEpisode:
        path = self.root / "episodes" / f"{episode_id}.jsonl"
        return ReplayEpisode.from_jsonl(episode_id, path)

    def list_episodes(self) -> list[str]:
        return sorted(path.stem for path in (self.root / "episodes").glob("*.jsonl"))

    def search(self, query: str, limit: int = 10) -> list[str]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []
        hits: list[str] = []
        for path in [self.root / "MEMORY.md", self.root / "FAILURES.md", self.root / "PROCEDURES.md"]:
            if not path.exists():
                continue
            for block in path.read_text(encoding="utf-8").split("§"):
                lower = block.lower()
                if all(term in lower for term in terms):
                    hits.append(block.strip())
                    if len(hits) >= limit:
                        return hits
        return hits

    def search_entries(
        self,
        query: str,
        *,
        namespace: Iterable[str] | None = None,
        category: str | None = None,
        tags: Iterable[str] = (),
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        return self.store.search(query, namespace=namespace, category=category, tags=tags, limit=limit, mode=mode)

    def consolidate_store(self, category: str | None = None, max_entries: int = 200) -> int:
        return self.store.consolidate(category=category, max_entries=max_entries)

    def consolidate_episode(self, episode_id: str, summary: dict) -> int:
        """Convert an episode summary into searchable durable memories."""
        created = 0
        status = str(summary.get("status", "UNKNOWN"))
        namespace = ("episodes", episode_id)
        self.add_fact(
            f"Episode {episode_id} ended with status {status} in {summary.get('steps', 0)} steps.",
            category="episode",
            namespace=namespace,
            tags=("episode", status.lower()),
            importance=0.5 if status != "WIN" else 0.7,
            source_episode_id=episode_id,
            metadata={"status": status, "steps": summary.get("steps", 0), "done": summary.get("done", False)},
        )
        created += 1
        for failure in summary.get("failures", []):
            self.add_failure(failure, source_episode_id=episode_id, metadata={"status": status})
            created += 1
        for hypothesis in summary.get("hypotheses", []):
            self.add_fact(
                hypothesis,
                category="rule",
                namespace=("rules",),
                tags=("hypothesis", "rule"),
                importance=0.65,
                confidence=0.6,
                source_episode_id=episode_id,
                metadata={"status": status},
            )
            created += 1
        for effect in summary.get("action_effects", []):
            if effect.get("changed_cells", 0) <= 0 and effect.get("reward", 0.0) <= 0:
                continue
            action = effect.get("action", {})
            text = (
                f"In episode {episode_id}, action {action.get('kind')} at {action.get('xy')} "
                f"changed {effect.get('changed_cells')} cells, reward={effect.get('reward')}, status={effect.get('status')}."
            )
            self.add_fact(
                text,
                category="insight",
                namespace=("action-effects",),
                tags=("action-effect", str(action.get("kind", "unknown")).lower()),
                importance=0.7 if effect.get("reward", 0.0) else 0.55,
                confidence=0.75,
                source_episode_id=episode_id,
                metadata=effect,
            )
            created += 1
        return created


@dataclass
class MemoryManager:
    """Facade that exposes working and durable memory."""

    durable: DurableMemory
    working: WorkingMemory = field(default_factory=WorkingMemory)

    def reset_episode(self) -> None:
        self.working.reset()

    def add_hypothesis(self, text: str) -> None:
        self.working.hypotheses.append(text)

    def add_note(self, text: str) -> None:
        self.working.notes.append(text)

    def add_fact(
        self,
        text: str,
        category: str = "fact",
        scope: str = "global",
        tags: Iterable[str] = (),
        namespace: Iterable[str] = ("global",),
        confidence: float = 1.0,
        importance: float = 0.5,
        source_episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.durable.add_fact(
            text,
            category=category,
            scope=scope,
            tags=tags,
            namespace=namespace,
            confidence=confidence,
            importance=importance,
            source_episode_id=source_episode_id,
            metadata=metadata,
        )

    def record_failure(self, text: str, durable: bool = True) -> None:
        self.working.failures.append(text)
        if durable:
            self.durable.add_failure(text)

    def search(self, query: str, limit: int = 10) -> list[str]:
        return self.durable.search(query, limit=limit)

    def search_entries(
        self,
        query: str,
        *,
        namespace: Iterable[str] | None = None,
        category: str | None = None,
        tags: Iterable[str] = (),
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        return self.durable.search_entries(query, namespace=namespace, category=category, tags=tags, limit=limit, mode=mode)

    def consolidate_episode(self, episode_id: str, summary: dict) -> int:
        return self.durable.consolidate_episode(episode_id, summary)


def _frame_to_json(frame: Frame) -> dict:
    return frame.to_dict(include_grid=True)


def _action_to_json(action: Action) -> dict:
    return action.to_dict()


def _step_to_json(record: StepRecord) -> dict:
    return {
        "type": "step",
        "step": record.step,
        "before": _frame_to_json(record.before),
        "action": _action_to_json(record.action),
        "after": _frame_to_json(record.after),
        "reward": record.reward,
        "changed_cells": record.changed_cells,
        "info": record.info,
    }


def _slugify(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip())
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "skill"
