"""Kaggle readiness checks for ARC-AGI-3 submissions."""

from __future__ import annotations

import importlib.util
import json
import shutil
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
    "capabilities.py",
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
    "loop_stages.py",
    "memory.py",
    "memory_policy.py",
    "memory_store.py",
    "models.py",
    "official.py",
    "official_eval.py",
    "policy.py",
    "replay.py",
    "recovery.py",
    "sandbox.py",
    "subagents.py",
    "submission.py",
    "thread.py",
    "tools.py",
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


@dataclass(frozen=True)
class KagglePackage:
    output_dir: str
    files: list[str]
    submission_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "files": self.files,
            "submission_path": self.submission_path,
            "manifest_path": self.manifest_path,
        }


def build_submission_manifest(package_root: str | Path = "arc_harness") -> list[str]:
    root = Path(package_root)
    return [str(root / name) for name in CORE_PACKAGE_FILES]


def build_kaggle_package(
    output_dir: str | Path,
    *,
    package_root: str | Path = "arc_harness",
    include_scripts: bool = True,
) -> KagglePackage:
    """Copy the harness into a Kaggle-ready directory.

    The output is deliberately plain files and folders, so it can be uploaded as
    a Kaggle Dataset or copied into a Notebook without a build step.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    package_root = Path(package_root)
    files: list[str] = []
    for source_text in build_submission_manifest(package_root):
        source = Path(source_text)
        target = output / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(str(target.relative_to(output)))

    submission_path = output / "submission.py"
    submission_path.write_text(_submission_template(), encoding="utf-8")
    files.append(str(submission_path.relative_to(output)))

    if include_scripts:
        script_source = Path("scripts") / "check_kaggle_readiness.py"
        if script_source.exists():
            script_target = output / script_source
            script_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(script_source, script_target)
            files.append(str(script_target.relative_to(output)))

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps({"files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
    return KagglePackage(str(output), files, str(submission_path), str(manifest_path))


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
    modules: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for name in ("arc_agi", "arcengine"):
        try:
            if importlib.util.find_spec(name) is None:
                raise ImportError(f"{name} not found")
            __import__(name)
            modules[name] = True
        except Exception as exc:
            modules[name] = False
            errors[name] = repr(exc)
    ok = all(modules.values())
    message = "official ARC-AGI-3 packages import successfully" if ok else "official packages not importable in this runtime"
    return ReadinessCheck("official_imports", ok, message, {"modules": modules, "errors": errors})


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


def _submission_template() -> str:
    return '''"""Kaggle ARC-AGI-3 submission entrypoint.

Upload this file next to the `arc_harness/` package directory, then expose
`is_done` and `choose_action` to the competition runtime.
"""

from arc_harness.submission import choose_action, is_done
'''
