"""Lazy exports for the utils package."""

from importlib import import_module
from typing import Any

_EXPORTS = {'ErrorContext': 'arc_harness.utils.errors', 'HarnessError': 'arc_harness.utils.errors', 'ValidationError': 'arc_harness.utils.errors', 'AgentEvent': 'arc_harness.utils.events', 'utc_now': 'arc_harness.utils.events', 'validate_action': 'arc_harness.utils.validation', 'validate_frame': 'arc_harness.utils.validation'}

__all__ = sorted(_EXPORTS)

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
