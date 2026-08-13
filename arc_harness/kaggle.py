"""Kaggle readiness checks for ARC-AGI-3 submissions."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import KaggleAgentAdapter
from .agent import ArcAgent, DelegatingPlannerAgent
from .models import load_model_from_config
from .official import EnvironmentFileCatalog


CORE_PACKAGE_FILES = (
    "actions.py",
    "adapters.py",
    "agent.py",
    "checkpoint.py",
    "config.py",
    "context.py",
    "delegation.py",
    "environment.py",
    "errors.py",
    "events.py",
    "evaluation.py",
    "guardrails.py",
    "hooks.py",
    "kaggle.py",
    "loop.py",
    "memory.py",
    "memory_policy.py",
    "memory_store.py",
    "models.py",
    "official.py",
    "official_eval.py",
    "policy.py",
    "replay.py",
    "subagents.py",
    "submission.py",
    "thread.py",
    "tracing.py",
    "validation.py",
    "__init__.py",
)


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "message": self.message, "metadata": self.metadata}


@dataclass(frozen=True)
class KaggleReadinessReport:
    checks: list[ReadinessCheck]
    manifest: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def failed(self) -> list[ReadinessCheck]:
        return [check for check in self.checks if not check.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "manifest": self.manifest,
        }


def build_submission_manifest(package_root: str | Path = "arc_harness") -> list[str]:
    root = Path(package_root)
    return [str(root / name) for name in CORE_PACKAGE_FILES]


def check_kaggle_readiness(
    *,
    package_root: str | Path = "arc_harness",
    environments_dir: str | Path | None = None,
    model_config: str | Path | None = None,
    agent: ArcAgent | None = None,
) -> KaggleReadinessReport:
    checks = [
        _check_package_files(package_root),
        _check_official_imports(),
        _check_environment_files(environments_dir) if environments_dir else ReadinessCheck("environment_files", True, "not configured"),
        _check_model_config(model_config) if model_config else ReadinessCheck("model_config", True, "not configured"),
        _check_submission_functions(agent or DelegatingPlannerAgent()),
    ]
    return KaggleReadinessReport(checks=checks, manifest=build_submission_manifest(package_root))


def _check_package_files(package_root: str | Path) -> ReadinessCheck:
    root = Path(package_root)
    missing = [path for path in build_submission_manifest(root) if not Path(path).exists()]
    return ReadinessCheck(
        "package_files",
        not missing,
        "all package files present" if not missing else f"missing {len(missing)} package files",
        {"missing": missing},
    )


def _check_official_imports() -> ReadinessCheck:
    modules = {name: importlib.util.find_spec(name) is not None for name in ("arc_agi", "arcengine")}
    ok = all(modules.values())
    message = "official ARC-AGI-3 packages available" if ok else "official packages not installed in this runtime"
    return ReadinessCheck("official_imports", ok, message, modules)


def _check_environment_files(environments_dir: str | Path) -> ReadinessCheck:
    catalog = EnvironmentFileCatalog(environments_dir)
    games = catalog.list_games()
    return ReadinessCheck(
        "environment_files",
        bool(games),
        f"found {len(games)} public games" if games else "no metadata.json files found",
        {"count": len(games), "sample": games[:3]},
    )


def _check_model_config(model_config: str | Path) -> ReadinessCheck:
    try:
        model = load_model_from_config(model_config)
        return ReadinessCheck("model_config", True, f"loaded model {model.name}", {"path": str(model_config)})
    except Exception as exc:
        return ReadinessCheck("model_config", False, f"failed to load model config: {exc}", {"path": str(model_config)})


def _check_submission_functions(agent: ArcAgent) -> ReadinessCheck:
    try:
        adapter = KaggleAgentAdapter(agent)
        action = adapter.choose_action([[[0, 1], [0, 2]]], [[0, 1], [0, 2]])
        done = adapter.is_done([[[0, 1], [0, 2]]], [[0, 1], [0, 2]])
        return ReadinessCheck("submission_functions", True, "choose_action/is_done smoke test passed", {"action": action, "done": done})
    except Exception as exc:
        return ReadinessCheck("submission_functions", False, f"submission smoke test failed: {exc}")
