"""Subagent delegation primitives for ARC harnesses.

The design intentionally favors manager-style delegation: the main agent keeps
control of the episode, while specialized subagents provide bounded, structured
analysis that can be traced and saved to memory.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from arc_harness.memory.context import ContextBundle
from arc_harness.utils.events import AgentEvent
from arc_harness.memory.memory import MemoryManager


@dataclass(frozen=True)
class SubTask:
    """A bounded task delegated to one specialist."""

    task_id: str
    kind: str
    payload: dict[str, Any]
    context: ContextBundle | None = None
    budget: int = 1000
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "context": self.context.to_dict() if self.context else None,
            "budget": self.budget,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SubAgentResult:
    """Structured result returned by a subagent."""

    task_id: str
    agent_name: str
    ok: bool
    output: dict[str, Any]
    summary: str
    confidence: float = 1.0
    trace: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "ok": self.ok,
            "output": self.output,
            "summary": self.summary,
            "confidence": self.confidence,
            "trace": self.trace,
            "metadata": self.metadata,
        }


class SubAgent(Protocol):
    """Protocol implemented by deterministic or model-backed specialists."""

    name: str
    kinds: tuple[str, ...]

    def run(self, task: SubTask, memory: MemoryManager) -> SubAgentResult:
        ...


class DelegationError(RuntimeError):
    """Raised when a subtask cannot be dispatched or completed."""


HandoffPredicate = Callable[[dict[str, Any], MemoryManager], bool]
EventSink = Callable[[AgentEvent], None]


@dataclass(frozen=True)
class DelegationConfig:
    """Runtime policy for subagent calls."""

    max_retries: int = 0
    remember_results: bool = True
    trace_events: bool = True
    raise_on_failure: bool = True
    parallel_workers: int = 4

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive.")


class DelegationManager:
    """Register and call subagents by task kind."""

    def __init__(
        self,
        subagents: Iterable[SubAgent] | None = None,
        *,
        config: DelegationConfig | None = None,
        remember_results: bool | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        base_config = config or DelegationConfig()
        if remember_results is not None:
            base_config = DelegationConfig(
                max_retries=base_config.max_retries,
                remember_results=remember_results,
                trace_events=base_config.trace_events,
                raise_on_failure=base_config.raise_on_failure,
                parallel_workers=base_config.parallel_workers,
            )
        self.config = base_config
        self.event_sink = event_sink
        self.events: list[AgentEvent] = []
        self._agents_by_name: dict[str, SubAgent] = {}
        self._agents_by_kind: dict[str, SubAgent] = {}
        for agent in subagents or []:
            self.register(agent)

    @classmethod
    def with_default_subagents(
        cls,
        *,
        config: DelegationConfig | None = None,
        remember_results: bool | None = None,
        event_sink: EventSink | None = None,
    ) -> "DelegationManager":
        from arc_harness.models.subagents import DiffSubAgent, ExplorerSubAgent, PerceptionSubAgent, PlannerSubAgent

        return cls(
            [PerceptionSubAgent(), DiffSubAgent(), ExplorerSubAgent(), PlannerSubAgent()],
            config=config,
            remember_results=remember_results,
            event_sink=event_sink,
        )

    def set_event_sink(self, event_sink: EventSink | None) -> None:
        self.event_sink = event_sink

    def register(self, agent: SubAgent) -> None:
        if agent.name in self._agents_by_name:
            raise DelegationError(f"Subagent {agent.name!r} is already registered.")
        if not agent.kinds:
            raise DelegationError(f"Subagent {agent.name!r} must support at least one task kind.")
        self._agents_by_name[agent.name] = agent
        for kind in agent.kinds:
            if kind in self._agents_by_kind:
                current = self._agents_by_kind[kind].name
                raise DelegationError(f"Task kind {kind!r} is already handled by {current!r}.")
            self._agents_by_kind[kind] = agent

    def available_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents_by_kind))

    def available_agents(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents_by_name))

    def delegate(
        self,
        kind: str,
        payload: dict[str, Any],
        memory: MemoryManager,
        *,
        context: ContextBundle | None = None,
        budget: int = 1000,
        metadata: dict[str, Any] | None = None,
        config: DelegationConfig | None = None,
    ) -> SubAgentResult:
        task = self._make_task(kind, payload, context=context, budget=budget, metadata=metadata)
        result = self._delegate_task(task, memory, config=config)
        if (config or self.config).remember_results:
            self._remember_result(task.kind, result, memory)
        return result

    def delegate_many(
        self,
        tasks: Iterable[SubTask | tuple[str, dict[str, Any]] | dict[str, Any]],
        memory: MemoryManager,
        *,
        config: DelegationConfig | None = None,
    ) -> list[SubAgentResult]:
        run_config = config or self.config
        subtasks = [self._coerce_task(task) for task in tasks]
        if not subtasks:
            return []

        max_workers = min(run_config.parallel_workers, len(subtasks))
        results_by_id: dict[str, SubAgentResult] = {}
        if max_workers <= 1:
            for task in subtasks:
                results_by_id[task.task_id] = self._delegate_task(task, memory, config=run_config)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._delegate_task, task, memory, run_config): task for task in subtasks}
                for future in as_completed(futures):
                    task = futures[future]
                    results_by_id[task.task_id] = future.result()

        results = [results_by_id[task.task_id] for task in subtasks]
        if run_config.remember_results:
            for task, result in zip(subtasks, results):
                self._remember_result(task.kind, result, memory)
        return results

    def _make_task(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        context: ContextBundle | None = None,
        budget: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> SubTask:
        return SubTask(
            task_id=str(uuid.uuid4()),
            kind=kind,
            payload=dict(payload),
            context=context,
            budget=budget,
            metadata=dict(metadata or {}),
        )

    def _coerce_task(self, value: SubTask | tuple[str, dict[str, Any]] | dict[str, Any]) -> SubTask:
        if isinstance(value, SubTask):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return self._make_task(value[0], value[1])
        if isinstance(value, dict):
            return self._make_task(
                str(value["kind"]),
                dict(value.get("payload", {})),
                budget=int(value.get("budget", 1000)),
                metadata=dict(value.get("metadata", {})),
            )
        raise TypeError(f"Cannot coerce {value!r} to SubTask.")

    def _delegate_task(self, task: SubTask, memory: MemoryManager, config: DelegationConfig | None = None) -> SubAgentResult:
        run_config = config or self.config
        agent = self._agents_by_kind.get(task.kind)
        if agent is None:
            raise DelegationError(f"No subagent registered for task kind {task.kind!r}.")

        attempts = run_config.max_retries + 1
        started = time.perf_counter()
        self._emit("subtask.started", task, {"agent_name": agent.name, "attempts_allowed": attempts}, run_config)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            attempt_started = time.perf_counter()
            try:
                result = agent.run(task, memory)
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                attempt_ms = round((time.perf_counter() - attempt_started) * 1000, 3)
                trace = {
                    "duration_ms": duration_ms,
                    "attempt_duration_ms": attempt_ms,
                    "attempt": attempt,
                    "attempts": attempt,
                    "kind": task.kind,
                    "agent_name": agent.name,
                }
                if result.trace:
                    trace.update(result.trace)
                completed = SubAgentResult(
                    task_id=result.task_id or task.task_id,
                    agent_name=result.agent_name or agent.name,
                    ok=result.ok,
                    output=result.output,
                    summary=result.summary,
                    confidence=result.confidence,
                    trace=trace,
                    metadata={**task.metadata, **result.metadata},
                )
                self._emit("subtask.completed", task, {"result": completed.to_dict()}, run_config)
                return completed
            except Exception as exc:
                last_error = exc
                self._emit(
                    "subtask.retrying" if attempt < attempts else "subtask.failed",
                    task,
                    {"agent_name": agent.name, "attempt": attempt, "error": repr(exc)},
                    run_config,
                )

        assert last_error is not None
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        memory.record_failure(f"Subagent {agent.name} failed on {task.kind}: {last_error}", durable=True)
        failed = SubAgentResult(
            task_id=task.task_id,
            agent_name=agent.name,
            ok=False,
            output={"error": repr(last_error)},
            summary=f"failed after {attempts} attempts: {last_error}",
            confidence=0.0,
            trace={"duration_ms": duration_ms, "attempts": attempts, "kind": task.kind, "agent_name": agent.name},
            metadata=task.metadata,
        )
        if run_config.raise_on_failure:
            raise DelegationError(f"Subagent {agent.name!r} failed on task kind {task.kind!r}: {last_error}") from last_error
        return failed

    def _emit(self, event_type: str, task: SubTask, payload: dict[str, Any], config: DelegationConfig) -> None:
        if not config.trace_events:
            return
        event = AgentEvent(
            event_type,
            task.task_id,
            {
                "task": {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "budget": task.budget,
                    "metadata": task.metadata,
                },
                **payload,
            },
        )
        self.events.append(event)
        if self.event_sink:
            self.event_sink(event)

    def _remember_result(self, kind: str, result: SubAgentResult, memory: MemoryManager) -> None:
        category = "insight" if result.ok else "failure"
        text = f"{result.agent_name} handled {kind}: {result.summary}"
        memory.add_fact(
            text,
            category=category,
            namespace=("delegation", kind),
            tags=("subagent", result.agent_name, kind),
            confidence=result.confidence,
            importance=0.55 if result.ok else 0.75,
            metadata=result.to_dict(),
        )


@dataclass(frozen=True)
class HandoffRule:
    """Route control to another agent when a state predicate matches."""

    target: str
    reason: str
    predicate: HandoffPredicate
    return_to_primary_on_done: bool = False

    def matches(self, state: dict[str, Any], memory: MemoryManager) -> bool:
        return self.predicate(state, memory)


@dataclass(frozen=True)
class HandoffRecord:
    """A traceable control-transfer event."""

    from_agent: str
    to_agent: str
    reason: str
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason,
            "step": self.step,
            "metadata": self.metadata,
        }


class HandoffController:
    """Stateful router for agent takeover inside one episode."""

    def __init__(self, primary_name: str, rules: Iterable[HandoffRule] | None = None) -> None:
        self.primary_name = primary_name
        self.active_name = primary_name
        self.rules = list(rules or [])
        self.records: list[HandoffRecord] = []

    def reset(self) -> None:
        self.active_name = self.primary_name
        self.records.clear()

    def choose_active(self, state: dict[str, Any], memory: MemoryManager) -> HandoffRecord | None:
        for rule in self.rules:
            if rule.target == self.active_name:
                continue
            if rule.matches(state, memory):
                record = HandoffRecord(
                    from_agent=self.active_name,
                    to_agent=rule.target,
                    reason=rule.reason,
                    step=int(state.get("step", 0)),
                    metadata={"return_to_primary_on_done": rule.return_to_primary_on_done},
                )
                self.active_name = rule.target
                self.records.append(record)
                memory.add_note(f"Handoff {record.from_agent} -> {record.to_agent}: {record.reason}")
                return record
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_name": self.primary_name,
            "active_name": self.active_name,
            "records": [record.to_dict() for record in self.records],
        }
