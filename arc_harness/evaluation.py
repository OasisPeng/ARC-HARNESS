"""Batch evaluation utilities for harness experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .agent import ArcAgent
from .config import RunnerConfig
from .environment import ArcEnvironment
from .loop import EpisodeResult
from .thread import ArcThread


EnvironmentFactory = Callable[[], ArcEnvironment]
AgentFactory = Callable[[], ArcAgent]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    env_factory: EnvironmentFactory
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    episode_id: str
    status: str
    done: bool
    steps: int
    total_reward: float
    failure_reason: str = "completed"
    trace_id: str | None = None
    replay_path: str | None = None
    trace_path: str | None = None
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "episode_id": self.episode_id,
            "status": self.status,
            "done": self.done,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "failure_reason": self.failure_reason,
            "trace_id": self.trace_id,
            "replay_path": self.replay_path,
            "trace_path": self.trace_path,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EvalReport:
    results: list[EvalCaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def completed(self) -> int:
        return sum(1 for result in self.results if result.done)

    @property
    def completion_rate(self) -> float:
        return self.completed / self.total if self.total else 0.0

    @property
    def average_steps(self) -> float:
        return sum(result.steps for result in self.results) / self.total if self.total else 0.0

    @property
    def average_success_steps(self) -> float:
        completed = [result.steps for result in self.results if result.done]
        return sum(completed) / len(completed) if completed else 0.0

    @property
    def average_reward(self) -> float:
        return sum(result.total_reward for result in self.results) / self.total if self.total else 0.0

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.failure_reason] = counts.get(result.failure_reason, 0) + 1
        return counts

    def by_case(self) -> dict[str, EvalCaseResult]:
        return {result.case_id: result for result in self.results}

    def compare(self, other: "EvalReport") -> dict:
        """Return headline deltas against another report."""
        return {
            "completion_rate_delta": self.completion_rate - other.completion_rate,
            "average_steps_delta": self.average_steps - other.average_steps,
            "average_success_steps_delta": self.average_success_steps - other.average_success_steps,
            "average_reward_delta": self.average_reward - other.average_reward,
            "status_counts": self.status_counts(),
            "baseline_status_counts": other.status_counts(),
            "failure_counts": self.failure_counts(),
            "baseline_failure_counts": other.failure_counts(),
        }

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "completion_rate": self.completion_rate,
            "average_steps": self.average_steps,
            "average_success_steps": self.average_success_steps,
            "average_reward": self.average_reward,
            "status_counts": self.status_counts(),
            "failure_counts": self.failure_counts(),
            "results": [result.to_dict() for result in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Report",
            "",
            f"- total: {self.total}",
            f"- completed: {self.completed}",
            f"- completion_rate: {self.completion_rate:.3f}",
            f"- average_steps: {self.average_steps:.3f}",
            f"- average_success_steps: {self.average_success_steps:.3f}",
            f"- average_reward: {self.average_reward:.3f}",
            f"- status_counts: {self.status_counts()}",
            f"- failure_counts: {self.failure_counts()}",
            "",
            "| case_id | status | done | steps | reward | failure_reason | trace_id |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for result in self.results:
            lines.append(
                f"| {result.case_id} | {result.status} | {result.done} | {result.steps} | "
                f"{result.total_reward:.3f} | {result.failure_reason} | {result.trace_id or ''} |"
            )
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


class EvaluationRunner:
    """Run an agent over multiple environment cases."""

    def __init__(self, memory_dir: str = ".arc_memory/evals") -> None:
        self.memory_dir = memory_dir

    def run(self, cases: Iterable[EvalCase], agent_factory: AgentFactory, config: RunnerConfig | None = None) -> EvalReport:
        results: list[EvalCaseResult] = []
        for case in cases:
            thread = ArcThread(memory_dir=self.memory_dir, metadata={"case_id": case.case_id, **case.metadata})
            episode = thread.run_episode(case.env_factory(), agent_factory(), config=config or RunnerConfig())
            trace_id = episode.summary.get("trace_id")
            results.append(
                EvalCaseResult(
                    case_id=case.case_id,
                    episode_id=episode.episode_id,
                    status=episode.status,
                    done=episode.done,
                    steps=episode.steps,
                    total_reward=episode.total_reward,
                    failure_reason=classify_failure(episode),
                    trace_id=trace_id,
                    replay_path=str(Path(self.memory_dir) / "memory" / "episodes" / f"{episode.episode_id}.jsonl"),
                    trace_path=str(Path(self.memory_dir) / "traces" / f"{trace_id}.json") if trace_id else None,
                    metrics=episode_metrics(episode),
                    metadata=case.metadata,
                )
            )
        return EvalReport(results)


def episode_metrics(episode: EpisodeResult) -> dict:
    effects = list(episode.summary.get("action_effects", []))
    noops = [effect for effect in effects if effect.get("changed_cells", 0) == 0 and effect.get("reward", 0.0) <= 0]
    changed = [effect for effect in effects if effect.get("changed_cells", 0) > 0]
    actions = [_action_key(effect.get("action", {})) for effect in effects]
    repeated_actions = len(actions) - len(set(actions))
    context_events = [event for event in episode.events if event.get("type") == "context.built"]
    dropped_sections = []
    context_tokens = []
    for event in context_events:
        context = event.get("payload", {}).get("context", {})
        dropped_sections.extend(context.get("dropped", []))
        if "total_tokens" in context:
            context_tokens.append(context["total_tokens"])
    return {
        "action_count": len(effects),
        "changed_action_count": len(changed),
        "noop_action_count": len(noops),
        "noop_rate": len(noops) / len(effects) if effects else 0.0,
        "unique_action_count": len(set(actions)),
        "repeated_action_count": repeated_actions,
        "context_event_count": len(context_events),
        "max_context_tokens": max(context_tokens) if context_tokens else 0,
        "dropped_context_sections": sorted(set(dropped_sections)),
    }


def classify_failure(episode: EpisodeResult) -> str:
    """Assign a stable, debuggable reason for evaluation grouping."""
    if episode.done or episode.status == "WIN":
        return "completed"
    summary = episode.summary
    error = summary.get("error") or {}
    error_text = f"{error.get('error_type', '')} {error.get('error', '')}".lower()
    failures = " ".join(str(item) for item in summary.get("failures", [])).lower()
    notes = " ".join(str(item) for item in summary.get("notes", [])).lower()
    combined = " ".join([error_text, failures, notes])
    if "permission" in combined or episode.status == "BLOCKED":
        return "permission_denied"
    if "subtask" in combined or "subagent" in combined:
        return "subagent_failed"
    if "guardrail" in combined:
        return "guardrail_failed"
    if "invalid action" in combined or "validationerror" in combined or "bad_action" in combined:
        return "invalid_action"
    if "planner" in combined and ("empty" in combined or "no plan" in combined):
        return "planner_empty"
    max_steps = summary.get("config", {}).get("max_steps")
    if isinstance(max_steps, int) and episode.steps >= max_steps:
        return "max_steps_exceeded"
    effects = list(summary.get("action_effects", []))
    if effects and all(effect.get("changed_cells", 0) == 0 and effect.get("reward", 0.0) <= 0 for effect in effects):
        actions = [_action_key(effect.get("action", {})) for effect in effects]
        if len(actions) != len(set(actions)):
            return "repeated_noop"
        return "state_not_progressing"
    if episode.status == "ERROR":
        return "runtime_error"
    return "unfinished"


def _action_key(action: dict) -> str:
    kind = str(action.get("kind", "UNKNOWN"))
    xy = action.get("xy")
    if xy is None:
        return kind
    return f"{kind}@{tuple(xy)}"
