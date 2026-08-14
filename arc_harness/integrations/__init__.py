"""Lazy exports for the integrations package."""

from importlib import import_module
from typing import Any

_EXPORTS = {'KaggleAgentAdapter': 'arc_harness.integrations.adapters', 'KagglePackage': 'arc_harness.integrations.kaggle', 'KaggleReadinessReport': 'arc_harness.integrations.kaggle', 'ReadinessCheck': 'arc_harness.integrations.kaggle', 'build_kaggle_package': 'arc_harness.integrations.kaggle', 'build_submission_manifest': 'arc_harness.integrations.kaggle', 'check_kaggle_readiness': 'arc_harness.integrations.kaggle', 'ArcAgi3Config': 'arc_harness.integrations.official', 'EnvironmentFileCatalog': 'arc_harness.integrations.official', 'OfficialArcEnvironment': 'arc_harness.integrations.official', 'OfficialDependencyError': 'arc_harness.integrations.official', 'coerce_official_frame': 'arc_harness.integrations.official', 'create_official_environment': 'arc_harness.integrations.official', 'resolve_official_action': 'arc_harness.integrations.official', 'OfficialSmokeReport': 'arc_harness.integrations.official_eval', 'OfficialSmokeResult': 'arc_harness.integrations.official_eval', 'OfficialSmokeRunner': 'arc_harness.integrations.official_eval', 'discover_official_games': 'arc_harness.integrations.official_eval', 'LocalSubprocessSandbox': 'arc_harness.integrations.sandbox', 'Sandbox': 'arc_harness.integrations.sandbox', 'SandboxCommand': 'arc_harness.integrations.sandbox', 'SandboxError': 'arc_harness.integrations.sandbox', 'SandboxPolicy': 'arc_harness.integrations.sandbox', 'SandboxPolicyError': 'arc_harness.integrations.sandbox', 'SandboxResult': 'arc_harness.integrations.sandbox'}

__all__ = sorted(_EXPORTS)

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
