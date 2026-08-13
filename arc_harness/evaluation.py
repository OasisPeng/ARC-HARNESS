"""Batch evaluation utilities for harness experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .agent import ArcAgent
from .config import RunnerConfig
from .environment import ArcEnvironment
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
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "episode_id": self.episode_id,
            "status": self.status,
            "done": self.done,
            "steps": self.steps,
            "total_reward": self.total_reward,
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

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "completion_rate": self.completion_rate,
            "average_steps": self.average_steps,
            "status_counts": self.status_counts(),
            "results": [result.to_dict() for result in self.results],
        }


class EvaluationRunner:
    """Run an agent over multiple environment cases."""

    def __init__(self, memory_dir: str = ".arc_memory/evals") -> None:
        self.memory_dir = memory_dir

    def run(self, cases: Iterable[EvalCase], agent_factory: AgentFactory, config: RunnerConfig | None = None) -> EvalReport:
        results: list[EvalCaseResult] = []
        for case in cases:
            thread = ArcThread(memory_dir=self.memory_dir, metadata={"case_id": case.case_id, **case.metadata})
            episode = thread.run_episode(case.env_factory(), agent_factory(), config=config or RunnerConfig())
            results.append(
                EvalCaseResult(
                    case_id=case.case_id,
                    episode_id=episode.episode_id,
                    status=episode.status,
                    done=episode.done,
                    steps=episode.steps,
                    total_reward=episode.total_reward,
                    metadata=case.metadata,
                )
            )
        return EvalReport(results)

