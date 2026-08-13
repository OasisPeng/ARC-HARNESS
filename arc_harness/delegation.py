"""Subagent delegation primitives for ARC harnesses.

The design intentionally favors manager-style delegation: the main agent keeps
control of the episode, while specialized subagents provide bounded, structured
analysis that can be traced and saved to memory.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from .context import ContextBundle
from .memory import MemoryManager


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


class DelegationManager:
    """Register and call subagents by task kind."""

    def __init__(self, subagents: Iterable[SubAgent] | None = None, *, remember_results: bool = True) -> None:
        self.remember_results = remember_results
        self._agents_by_name: dict[str, SubAgent] = {}
        self._agents_by_kind: dict[str, SubAgent] = {}
        for agent in subagents or []:
            self.register(agent)

    @classmethod
    def with_default_subagents(cls, *, remember_results: bool = True) -> "DelegationManager":
        from .subagents import DiffSubAgent, ExplorerSubAgent, PerceptionSubAgent

        return cls(
            [PerceptionSubAgent(), DiffSubAgent(), ExplorerSubAgent()],
            remember_results=remember_results,
        )

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
    ) -> SubAgentResult:
        agent = self._agents_by_kind.get(kind)
        if agent is None:
            raise DelegationError(f"No subagent registered for task kind {kind!r}.")

        task = SubTask(
            task_id=str(uuid.uuid4()),
            kind=kind,
            payload=dict(payload),
            context=context,
            budget=budget,
            metadata=dict(metadata or {}),
        )
        started = time.perf_counter()
        try:
            result = agent.run(task, memory)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            memory.record_failure(f"Subagent {agent.name} failed on {kind}: {exc}", durable=True)
            raise DelegationError(f"Subagent {agent.name!r} failed on task kind {kind!r}: {exc}") from exc

        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        trace = {"duration_ms": duration_ms, "kind": kind, "agent_name": agent.name}
        if result.trace:
            trace.update(result.trace)
        result = SubAgentResult(
            task_id=result.task_id or task.task_id,
            agent_name=result.agent_name or agent.name,
            ok=result.ok,
            output=result.output,
            summary=result.summary,
            confidence=result.confidence,
            trace=trace,
            metadata={**task.metadata, **result.metadata},
        )
        if self.remember_results:
            self._remember_result(kind, result, memory)
        return result

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
