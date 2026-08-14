"""Codex SDK-inspired Thread API for ARC experiments."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .agent import ArcAgent
from .checkpoint import CheckpointStore
from .config import RunnerConfig
from .context import ContextBundle, ContextInjector, ContextManager
from .delegation import DelegationConfig, DelegationManager, SubAgentResult, SubTask
from .environment import ArcEnvironment
from .events import AgentEvent, utc_now
from .hooks import Hook, HookManager
from .loop import EpisodeResult, EpisodeRunner
from .loop_stages import StagePipeline
from .memory import DurableMemory, MemoryManager
from .tracing import Trace, TraceStore


@dataclass
class ArcThread:
    """A resumable experiment thread.

    A thread owns memory, hooks, and a run history. It intentionally mirrors the
    ergonomic shape of Codex's start/resume/run API without depending on Codex.
    """

    memory_dir: str | Path = ".arc_memory"
    thread_id: str | None = None
    hooks: list[Hook] = field(default_factory=list)
    guardrails: list = field(default_factory=list)
    delegation: DelegationManager | None = None
    pipeline: StagePipeline | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.thread_id = self.thread_id or str(uuid.uuid4())
        self.root = Path(self.memory_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "threads" / f"{self.thread_id}.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryManager(DurableMemory(self.root / "memory"))
        self.delegation_events: list[dict] = []
        self.delegation_trace = Trace("ARC delegation", group_id=self.thread_id, metadata={"thread_id": self.thread_id})
        self._delegation_spans = {}
        self.delegation = self.delegation or DelegationManager.with_default_subagents()
        self.delegation.set_event_sink(self._on_delegation_event)
        self.history: list[dict] = []
        self.created_at = utc_now()
        self.updated_at = self.created_at
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.history = data.get("history", [])
            self.delegation_events = data.get("delegation_events", [])
            self.metadata = data.get("metadata", self.metadata)
            self.created_at = data.get("created_at", self.created_at)
            self.updated_at = data.get("updated_at", self.updated_at)
        else:
            self._save_state()

    @classmethod
    def resume(cls, thread_id: str, memory_dir: str | Path = ".arc_memory", hooks: list[Hook] | None = None) -> "ArcThread":
        return cls(memory_dir=memory_dir, thread_id=thread_id, hooks=list(hooks or []))

    @classmethod
    def list_threads(cls, memory_dir: str | Path = ".arc_memory") -> list[dict]:
        thread_dir = Path(memory_dir) / "threads"
        if not thread_dir.exists():
            return []
        threads = []
        for path in sorted(thread_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            threads.append({
                "thread_id": data["thread_id"],
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "metadata": data.get("metadata", {}),
                "runs": len(data.get("history", [])),
            })
        return threads

    @classmethod
    def read_thread(cls, thread_id: str, memory_dir: str | Path = ".arc_memory") -> dict:
        path = Path(memory_dir) / "threads" / f"{thread_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def run_episode(
        self,
        env: ArcEnvironment,
        agent: ArcAgent,
        max_steps: int = 256,
        config: RunnerConfig | None = None,
        pipeline: StagePipeline | None = None,
    ) -> EpisodeResult:
        run_config = config or RunnerConfig.from_value(max_steps)
        runner = EpisodeRunner(
            self.memory,
            HookManager(self.hooks),
            CheckpointStore(self.root / "checkpoints", enabled=run_config.checkpoint),
            TraceStore(self.root / "traces", enabled=run_config.tracing),
            self.guardrails,
            pipeline=pipeline or self.pipeline,
            delegation=self.delegation,
        )
        result = runner.run(env, agent, config=run_config)
        self.history.append(result.to_dict())
        self._save_state()
        return result

    def run_streamed(
        self,
        env: ArcEnvironment,
        agent: ArcAgent,
        max_steps: int = 256,
        config: RunnerConfig | None = None,
        pipeline: StagePipeline | None = None,
    ) -> Iterator[dict]:
        run_config = config or RunnerConfig.from_value(max_steps)
        runner = EpisodeRunner(
            self.memory,
            HookManager(self.hooks),
            CheckpointStore(self.root / "checkpoints", enabled=run_config.checkpoint),
            TraceStore(self.root / "traces", enabled=run_config.tracing),
            self.guardrails,
            pipeline=pipeline or self.pipeline,
            delegation=self.delegation,
        )
        completed: EpisodeResult | None = None
        event_dicts: list[dict] = []
        for event in runner.run_events(env, agent, config=run_config):
            payload = event.to_dict()
            payload["thread_id"] = self.thread_id
            event_dicts.append(payload)
            if event.type == "episode.completed":
                completed = event.payload["result"]
                completed.events = event_dicts
            yield payload
        if completed is not None:
            self.history.append(completed.to_dict())
            self._save_state()

    def add_metadata(self, **metadata: str) -> None:
        self.metadata.update(metadata)
        self._save_state()

    def read_episode(self, episode_id: str) -> list[dict]:
        return self.memory.durable.read_episode(episode_id)

    def load_replay(self, episode_id: str):
        return self.memory.durable.load_replay(episode_id)

    def read_checkpoint(self, episode_id: str) -> dict:
        return CheckpointStore(self.root / "checkpoints").read(episode_id)

    def latest_checkpoint(self) -> dict | None:
        return CheckpointStore(self.root / "checkpoints").latest()

    def read_trace(self, trace_id: str) -> dict:
        return TraceStore(self.root / "traces").read(trace_id)

    def build_context(
        self,
        *,
        latest_frame=None,
        trace_id: str | None = None,
        query: str = "",
        manager: ContextManager | None = None,
    ) -> ContextBundle:
        trace = self.read_trace(trace_id) if trace_id else None
        context_manager = manager or ContextManager()
        return context_manager.build(memory=self.memory, latest_frame=latest_frame, trace=trace, query=query)

    def inject_context(self, **kwargs) -> str:
        return ContextInjector().inject(memory=self.memory, **kwargs)

    def delegate(
        self,
        kind: str,
        payload: dict,
        *,
        context: ContextBundle | None = None,
        budget: int = 1000,
        metadata: dict | None = None,
        config: DelegationConfig | None = None,
    ) -> SubAgentResult:
        result = self.delegation.delegate(
            kind,
            payload,
            self.memory,
            context=context,
            budget=budget,
            metadata=metadata,
            config=config,
        )
        self._save_state()
        return result

    def delegate_many(
        self,
        tasks: list[SubTask | tuple[str, dict] | dict],
        *,
        config: DelegationConfig | None = None,
    ) -> list[SubAgentResult]:
        results = self.delegation.delegate_many(tasks, self.memory, config=config)
        self._save_state()
        return results

    def available_subtasks(self) -> tuple[str, ...]:
        return self.delegation.available_kinds()

    def read_delegation_events(self) -> list[dict]:
        return list(self.delegation_events)

    def read_delegation_trace(self) -> dict:
        return TraceStore(self.root / "traces").read(self.delegation_trace.trace_id)

    def _on_delegation_event(self, event: AgentEvent) -> None:
        payload = event.to_dict()
        payload["thread_id"] = self.thread_id
        self.delegation_events.append(payload)
        task = payload.get("payload", {}).get("task", {})
        task_id = task.get("task_id")
        if not task_id:
            return
        if event.type == "subtask.started":
            self._delegation_spans[task_id] = self.delegation_trace.start_span(
                f"subagent.{task.get('kind')}",
                metadata={
                    "task_id": task_id,
                    "kind": task.get("kind"),
                    "agent_name": payload.get("payload", {}).get("agent_name"),
                    "budget": task.get("budget"),
                    "metadata": task.get("metadata", {}),
                },
            )
            return
        if event.type in {"subtask.completed", "subtask.failed"}:
            span = self._delegation_spans.pop(task_id, None)
            if span is not None:
                span.finish({"event": event.type, **payload.get("payload", {})})
            self.delegation_trace.finish({"last_event": event.type, "event_count": len(self.delegation_events)})
            store = TraceStore(self.root / "traces")
            store.write(self.delegation_trace)
            store.append_jsonl(self.delegation_trace)

    def _save_state(self) -> None:
        self.updated_at = utc_now()
        payload = {
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "history": self.history,
            "delegation_events": self.delegation_events[-500:],
            "delegation_trace_id": self.delegation_trace.trace_id,
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
