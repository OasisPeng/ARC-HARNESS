"""Context management and injection for ARC agents.

The design borrows from:
- OpenAI Agents SDK: separate local runtime context from model-visible context.
- LangGraph: trim/summarize state before model calls.
- Deep Agents: offload bulky state and inject compact references.
- Claude Code: re-inject stable instructions/memory after compaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from .actions import Frame, StepRecord
from .memory import MemoryManager
from .memory_policy import MemoryPolicy


TokenCounter = Callable[[str], int]


class ContextRole(str, Enum):
    POLICY = "policy"
    MEMORY = "memory"
    RECENT_STEPS = "recent_steps"
    TRACE = "trace"
    FRAME = "frame"
    NOTES = "notes"
    OFFLOADED = "offloaded"


@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 1200
    memory_tokens: int = 360
    recent_step_tokens: int = 360
    trace_tokens: int = 220
    frame_tokens: int = 180
    notes_tokens: int = 120

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")


@dataclass(frozen=True)
class ContextSection:
    role: ContextRole
    title: str
    content: str
    priority: int = 100
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        if not self.content.strip():
            return ""
        return f"<{self.role.value} title=\"{self.title}\">\n{self.content.strip()}\n</{self.role.value}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "title": self.title,
            "content": self.content,
            "priority": self.priority,
            "tokens": self.tokens,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ContextBundle:
    sections: list[ContextSection]
    total_tokens: int
    budget: ContextBudget
    dropped: list[ContextSection] = field(default_factory=list)

    def render(self) -> str:
        return "\n\n".join(section.render() for section in self.sections if section.render())

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "budget": self.budget.__dict__,
            "sections": [section.to_dict() for section in self.sections],
            "dropped": [section.to_dict() for section in self.dropped],
        }


class ContextManager:
    """Build compact, model-visible context from memory, steps, frame, and trace."""

    def __init__(self, budget: ContextBudget | None = None, token_counter: TokenCounter | None = None) -> None:
        self.budget = budget or ContextBudget()
        self.token_counter = token_counter or approximate_tokens

    def build(
        self,
        *,
        memory: MemoryManager,
        latest_frame: Frame | None = None,
        trace: dict[str, Any] | None = None,
        query: str = "",
        include_policy: bool = True,
        memory_limit: int = 6,
        recent_step_limit: int = 8,
    ) -> ContextBundle:
        sections: list[ContextSection] = []
        if include_policy:
            sections.append(self._section(ContextRole.POLICY, "Memory Policy", MemoryPolicy().render(), 10, self.budget.notes_tokens))
        if query:
            sections.append(self._memory_section(memory, query, memory_limit))
        sections.append(self._recent_steps_section(memory.working.steps[-recent_step_limit:]))
        if latest_frame is not None:
            sections.append(self._frame_section(latest_frame))
        if trace:
            sections.append(self._trace_section(trace))
        if memory.working.notes or memory.working.hypotheses or memory.working.failures:
            sections.append(self._notes_section(memory))
        return self._fit_budget([section for section in sections if section.content.strip()])

    def _memory_section(self, memory: MemoryManager, query: str, limit: int) -> ContextSection:
        hits = memory.search_entries(query, limit=limit, mode="hybrid")
        lines = []
        for idx, hit in enumerate(hits, 1):
            tags = ",".join(hit.tags)
            lines.append(
                f"{idx}. [{hit.category} score={hit.score:.3f} ns={'/'.join(hit.namespace)} tags={tags}] {hit.text}"
            )
        content = "\n".join(lines) if lines else "No durable memory hits."
        return self._section(ContextRole.MEMORY, "Relevant Durable Memory", content, 20, self.budget.memory_tokens, {"query": query})

    def _recent_steps_section(self, records: Iterable[StepRecord]) -> ContextSection:
        lines = []
        for record in records:
            action = record.action.to_competition_value()
            lines.append(
                f"step={record.step} action={action} changed={record.changed_cells} reward={record.reward} status={record.after.status}"
            )
        content = "\n".join(lines) if lines else "No recent steps yet."
        return self._section(ContextRole.RECENT_STEPS, "Recent Action Effects", content, 30, self.budget.recent_step_tokens)

    def _frame_section(self, frame: Frame) -> ContextSection:
        colors = sorted({cell for row in frame.grid for cell in row})
        content = f"status={frame.status} size={frame.width}x{frame.height} colors={colors}"
        return self._section(ContextRole.FRAME, "Latest Frame Summary", content, 25, self.budget.frame_tokens)

    def _trace_section(self, trace: dict[str, Any]) -> ContextSection:
        spans = trace.get("spans", [])
        lines = [
            f"trace_id={trace.get('trace_id')} workflow={trace.get('workflow_name')} status={trace.get('metadata', {}).get('status')}"
        ]
        for span in spans[-8:]:
            lines.append(
                f"span={span.get('name')} duration_ms={_round(span.get('duration_ms'))} metadata={_short_json(span.get('metadata', {}), 160)}"
            )
        return self._section(ContextRole.TRACE, "Trace Summary", "\n".join(lines), 28, self.budget.trace_tokens)

    def _notes_section(self, memory: MemoryManager) -> ContextSection:
        parts = []
        if memory.working.hypotheses:
            parts.append("Hypotheses: " + " | ".join(memory.working.hypotheses[-5:]))
        if memory.working.failures:
            parts.append("Failures: " + " | ".join(memory.working.failures[-5:]))
        if memory.working.notes:
            parts.append("Notes: " + " | ".join(memory.working.notes[-5:]))
        return self._section(ContextRole.NOTES, "Working Notes", "\n".join(parts), 35, self.budget.notes_tokens)

    def _section(
        self,
        role: ContextRole,
        title: str,
        content: str,
        priority: int,
        max_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> ContextSection:
        trimmed = trim_to_tokens(content, max_tokens, self.token_counter)
        return ContextSection(
            role=role,
            title=title,
            content=trimmed,
            priority=priority,
            tokens=self.token_counter(trimmed),
            metadata=dict(metadata or {}),
        )

    def _fit_budget(self, sections: list[ContextSection]) -> ContextBundle:
        kept: list[ContextSection] = []
        dropped: list[ContextSection] = []
        total = 0
        for section in sorted(sections, key=lambda item: item.priority):
            if total + section.tokens <= self.budget.max_tokens:
                kept.append(section)
                total += section.tokens
            else:
                dropped.append(section)
        kept.sort(key=lambda item: item.priority)
        return ContextBundle(kept, total, self.budget, dropped)


class ContextInjector:
    """Small helper that builds context and returns prompt-ready text."""

    def __init__(self, manager: ContextManager | None = None) -> None:
        self.manager = manager or ContextManager()

    def inject(self, **kwargs: Any) -> str:
        return self.manager.build(**kwargs).render()


def approximate_tokens(text: str) -> int:
    # Cheap model-agnostic approximation: good enough for deterministic tests
    # and offline Kaggle use. Replace with model-specific tokenizers later.
    return max(1, (len(text) + 3) // 4) if text else 0


def trim_to_tokens(text: str, max_tokens: int, token_counter: TokenCounter = approximate_tokens) -> str:
    if token_counter(text) <= max_tokens:
        return text
    words = text.split()
    result: list[str] = []
    for word in words:
        candidate = " ".join([*result, word])
        if token_counter(candidate) > max_tokens:
            break
        result.append(word)
    if result:
        return " ".join(result) + "\n[trimmed]"
    max_chars = max_tokens * 4
    return text[:max_chars].rstrip() + "\n[trimmed]"


def _short_json(value: Any, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + "...[trimmed]"


def _round(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value
