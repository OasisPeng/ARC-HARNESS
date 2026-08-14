"""Lazy exports for the core package."""

from importlib import import_module
from typing import Any

_EXPORTS = {'Action': 'arc_harness.core.actions', 'ActionType': 'arc_harness.core.actions', 'Frame': 'arc_harness.core.actions', 'StepRecord': 'arc_harness.core.actions', 'ArcAgent': 'arc_harness.core.agent', 'DelegatingPlannerAgent': 'arc_harness.core.agent', 'HandoffAgent': 'arc_harness.core.agent', 'HeuristicAgent': 'arc_harness.core.agent', 'RuleLearningAgent': 'arc_harness.core.agent', 'RunnerConfig': 'arc_harness.core.config', 'ArcEnvironment': 'arc_harness.core.environment', 'EnvironmentResult': 'arc_harness.core.environment', 'validate_environment': 'arc_harness.core.environment', 'Decision': 'arc_harness.core.policy', 'HookDecision': 'arc_harness.core.policy'}

__all__ = sorted(_EXPORTS)

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
