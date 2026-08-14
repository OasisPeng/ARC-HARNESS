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

from arc_harness.core.actions import Action, Frame, StepRecord
from arc_harness.memory.memory import MemoryManager
from arc_harness.memory.memory_policy import MemoryPolicy


TokenCounter = Callable[[str], int]


class ContextRole(str, Enum):
    POLICY = "policy"
    MEMORY = "memory"
    RECENT_STEPS = "recent_steps"
    TRACE = "trace"
    FRAME = "frame"
    OBJECTS = "objects"
    ACTION_MAP = "action_map"
    PLAN = "plan"
    RECOVERY = "recovery"
    NOTES = "notes"
    OFFLOADED = "offloaded"


@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int = 1200
    memory_tokens: int = 360
    recent_step_tokens: int = 360
    trace_tokens: int = 220
    frame_tokens: int = 180
    object_tokens: int = 220
    action_map_tokens: int = 260
    plan_tokens: int = 180
    recovery_tokens: int = 140
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
        include_arc_state: bool = True,
        current_plan: dict[str, Any] | None = None,
    ) -> ContextBundle:
        sections: list[ContextSection] = []
        if include_policy:
            sections.append(self._section(ContextRole.POLICY, "Memory Policy", MemoryPolicy().render(), 10, self.budget.notes_tokens))
        if query:
            sections.append(self._memory_section(memory, query, memory_limit))
        sections.append(self._recent_steps_section(memory.working.steps[-recent_step_limit:]))
        if latest_frame is not None:
            sections.append(self._frame_section(latest_frame))
            if include_arc_state:
                sections.append(self._object_section(latest_frame))
        if include_arc_state:
            sections.append(self._action_map_section(memory.working.steps))
            if current_plan:
                sections.append(self._plan_section(current_plan))
            if memory.working.notes or memory.working.failures:
                sections.append(self._recovery_section(memory))
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
        return self._section(ContextRole.RECENT_STEPS, "Recent Action Effects", content, 21, self.budget.recent_step_tokens)

    def _frame_section(self, frame: Frame) -> ContextSection:
        colors = sorted({cell for row in frame.grid for cell in row})
        content = f"status={frame.status} size={frame.width}x{frame.height} colors={colors}"
        return self._section(ContextRole.FRAME, "Latest Frame Summary", content, 25, self.budget.frame_tokens)

    def _object_section(self, frame: Frame) -> ContextSection:
        components = _connected_components(frame)
        lines = [
            f"object_count={len(components)} nonzero_objects={len([item for item in components if item['color'] != 0])}"
        ]
        for item in components[:8]:
            lines.append(
                f"color={item['color']} size={item['size']} bbox={item['bbox']} center={item['center']} sample={item['sample_cells'][:4]}"
            )
        return self._section(
            ContextRole.OBJECTS,
            "Object Summary",
            "\n".join(lines),
            24,
            self.budget.object_tokens,
            {"component_count": len(components)},
        )

    def _action_map_section(self, records: Iterable[StepRecord]) -> ContextSection:
        tried: list[str] = []
        changed: list[str] = []
        failed: list[str] = []
        by_action: dict[str, dict[str, Any]] = {}
        for record in records:
            key = _action_text(record.action)
            entry = by_action.setdefault(key, {"tries": 0, "changed": 0, "reward": 0.0, "statuses": []})
            entry["tries"] += 1
            entry["changed"] += record.changed_cells
            entry["reward"] += record.reward
            entry["statuses"].append(record.after.status)
        for key, entry in sorted(by_action.items()):
            text = (
                f"{key}: tries={entry['tries']} changed={entry['changed']} "
                f"reward={round(entry['reward'], 3)} last_status={entry['statuses'][-1]}"
            )
            tried.append(text)
            if entry["changed"] > 0 or entry["reward"] > 0:
                changed.append(text)
            else:
                failed.append(text)
        lines = [
            "tried:",
            *(_limit_lines(tried, 10) or ["none"]),
            "effective:",
            *(_limit_lines(changed, 6) or ["none"]),
            "failed_or_noop:",
            *(_limit_lines(failed, 10) or ["none"]),
        ]
        return self._section(
            ContextRole.ACTION_MAP,
            "Tried And Failed Action Map",
            "\n".join(lines),
            22,
            self.budget.action_map_tokens,
            {"tried_count": len(tried), "effective_count": len(changed), "failed_count": len(failed)},
        )

    def _plan_section(self, plan: dict[str, Any]) -> ContextSection:
        steps = plan.get("plan", []) if isinstance(plan, dict) else []
        lines = [
            f"candidate_count={plan.get('candidate_count')} planned_count={plan.get('planned_count')} stop_reason={plan.get('stop_reason')}"
        ]
        for idx, step in enumerate(steps[:5], 1):
            lines.append(
                f"{idx}. action={step.get('action')} score={step.get('score')} reason={step.get('reason')}"
            )
        return self._section(ContextRole.PLAN, "Current Plan Summary", "\n".join(lines), 26, self.budget.plan_tokens)

    def _recovery_section(self, memory: MemoryManager) -> ContextSection:
        recovery_notes = [note for note in memory.working.notes if "Recovery " in note or "recovery" in note.lower()]
        recent_failures = list(memory.working.failures[-5:])
        if not recovery_notes and not recent_failures:
            return self._section(ContextRole.RECOVERY, "Recovery Summary", "No recovery events yet.", 34, self.budget.recovery_tokens)
        lines = []
        if recovery_notes:
            lines.append("recovery_notes: " + " | ".join(recovery_notes[-5:]))
        if recent_failures:
            lines.append("recent_failures: " + " | ".join(recent_failures))
        return self._section(ContextRole.RECOVERY, "Recovery Summary", "\n".join(lines), 34, self.budget.recovery_tokens)

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


def _action_text(action: Action) -> str:
    value = action.to_competition_value()
    if isinstance(value, tuple):
        return f"{value[0]}({value[1]},{value[2]})"
    return str(value)


def _limit_lines(lines: list[str], limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    return [*lines[:limit], f"... {len(lines) - limit} more"]


def _connected_components(frame: Frame) -> list[dict[str, Any]]:
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y, row in enumerate(frame.grid):
        for x, color in enumerate(row):
            if (x, y) in visited:
                continue
            cells = _flood_fill(frame, x, y, color, visited)
            bbox = _bbox(cells)
            center = ((bbox["x1"] + bbox["x2"]) // 2, (bbox["y1"] + bbox["y2"]) // 2)
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": bbox,
                    "center": center,
                    "sample_cells": cells[:8],
                }
            )
    components.sort(key=lambda item: (item["color"] == 0, -item["size"], item["color"], item["bbox"]["y1"], item["bbox"]["x1"]))
    return components


def _flood_fill(frame: Frame, start_x: int, start_y: int, color: int, visited: set[tuple[int, int]]) -> list[tuple[int, int]]:
    stack = [(start_x, start_y)]
    visited.add((start_x, start_y))
    cells: list[tuple[int, int]] = []
    while stack:
        x, y = stack.pop()
        cells.append((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or ny >= frame.height or nx >= frame.width:
                continue
            if (nx, ny) in visited or frame.grid[ny][nx] != color:
                continue
            visited.add((nx, ny))
            stack.append((nx, ny))
    return cells


def _bbox(cells: list[tuple[int, int]]) -> dict[str, int]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
