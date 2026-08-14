"""Lazy exports for the eval package."""

from importlib import import_module
from typing import Any

_EXPORTS = {'EvalCase': 'arc_harness.eval.evaluation', 'EvalCaseResult': 'arc_harness.eval.evaluation', 'EvalReport': 'arc_harness.eval.evaluation', 'EvaluationRunner': 'arc_harness.eval.evaluation', 'classify_failure': 'arc_harness.eval.evaluation', 'episode_metrics': 'arc_harness.eval.evaluation', 'ReplayEpisode': 'arc_harness.eval.replay', 'ReplayStep': 'arc_harness.eval.replay', 'Span': 'arc_harness.eval.tracing', 'Trace': 'arc_harness.eval.tracing', 'TraceStore': 'arc_harness.eval.tracing', 'TraceTimeline': 'arc_harness.eval.tracing', 'TraceTimelineItem': 'arc_harness.eval.tracing'}

__all__ = sorted(_EXPORTS)

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
