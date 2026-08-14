"""Smoke evaluation helpers for public ARC-AGI-3 environment files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from arc_harness.core.agent import ArcAgent
from arc_harness.core.config import RunnerConfig
from arc_harness.integrations.official import ArcAgi3Config, EnvironmentFileCatalog, OfficialDependencyError, create_official_environment
from arc_harness.runtime.thread import ArcThread


@dataclass(frozen=True)
class OfficialSmokeResult:
    game_id: str
    status: str
    done: bool
    steps: int
    episode_id: str | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "status": self.status,
            "done": self.done,
            "steps": self.steps,
            "episode_id": self.episode_id,
            "error": self.error,
        }


@dataclass(frozen=True)
class OfficialSmokeReport:
    results: list[OfficialSmokeResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def runnable(self) -> int:
        return sum(1 for result in self.results if not result.error)

    @property
    def completed(self) -> int:
        return sum(1 for result in self.results if result.done)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "runnable": self.runnable,
            "completed": self.completed,
            "results": [result.to_dict() for result in self.results],
        }


class OfficialSmokeRunner:
    """Run a bounded harness episode across discovered official games."""

    def __init__(
        self,
        environments_dir: str | Path = "environment_files",
        *,
        memory_dir: str | Path = ".arc_memory/official_smoke",
        operation_mode: str = "OFFLINE",
    ) -> None:
        self.environments_dir = Path(environments_dir)
        self.memory_dir = Path(memory_dir)
        self.operation_mode = operation_mode

    def run(self, agent: ArcAgent, *, max_games: int | None = None, config: RunnerConfig | None = None) -> OfficialSmokeReport:
        games = EnvironmentFileCatalog(self.environments_dir).list_games()
        selected = games[:max_games] if max_games is not None else games
        results: list[OfficialSmokeResult] = []
        for game in selected:
            game_id = game["game_id"]
            try:
                env = create_official_environment(
                    ArcAgi3Config(
                        game_id=game_id,
                        operation_mode=self.operation_mode,
                        environments_dir=self.environments_dir,
                    )
                )
                thread = ArcThread(memory_dir=self.memory_dir, metadata={"game_id": game_id})
                episode = thread.run_episode(env, agent, config=config or RunnerConfig(max_steps=32, abort_on_error=False))
                results.append(OfficialSmokeResult(game_id, episode.status, episode.done, episode.steps, episode.episode_id))
            except OfficialDependencyError as exc:
                results.append(OfficialSmokeResult(game_id, "SKIPPED", False, 0, error=str(exc)))
            except Exception as exc:
                results.append(OfficialSmokeResult(game_id, "ERROR", False, 0, error=repr(exc)))
        return OfficialSmokeReport(results)


def discover_official_games(environments_dir: str | Path = "environment_files") -> list[dict]:
    return EnvironmentFileCatalog(environments_dir).list_games()
