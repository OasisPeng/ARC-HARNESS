"""Lazy exports for the runtime package."""

from importlib import import_module
from typing import Any

_EXPORTS = {'CheckpointStore': 'arc_harness.runtime.checkpoint', 'ActionGuardrail': 'arc_harness.runtime.guardrails', 'CoordinateBoundsGuardrail': 'arc_harness.runtime.guardrails', 'FrameGuardrail': 'arc_harness.runtime.guardrails', 'GuardrailDecision': 'arc_harness.runtime.guardrails', 'GuardrailResult': 'arc_harness.runtime.guardrails', 'MaxChangedCellsGuardrail': 'arc_harness.runtime.guardrails', 'ResultGuardrail': 'arc_harness.runtime.guardrails', 'ActionBudgetHook': 'arc_harness.runtime.hooks', 'Hook': 'arc_harness.runtime.hooks', 'HookManager': 'arc_harness.runtime.hooks', 'HookMatcher': 'arc_harness.runtime.hooks', 'JsonlTraceHook': 'arc_harness.runtime.hooks', 'EpisodeResult': 'arc_harness.runtime.loop', 'EpisodeRunner': 'arc_harness.runtime.loop', 'ActionExecutionStage': 'arc_harness.runtime.loop_stages', 'BuildContextStage': 'arc_harness.runtime.loop_stages', 'DecisionStage': 'arc_harness.runtime.loop_stages', 'DoneCheckStage': 'arc_harness.runtime.loop_stages', 'ExplorationStage': 'arc_harness.runtime.loop_stages', 'LoopRuntime': 'arc_harness.runtime.loop_stages', 'LoopStage': 'arc_harness.runtime.loop_stages', 'LoopState': 'arc_harness.runtime.loop_stages', 'PerceptionStage': 'arc_harness.runtime.loop_stages', 'PermissionStage': 'arc_harness.runtime.loop_stages', 'PlanDecisionStage': 'arc_harness.runtime.loop_stages', 'PlanningStage': 'arc_harness.runtime.loop_stages', 'StagePipeline': 'arc_harness.runtime.loop_stages', 'StopCheckStage': 'arc_harness.runtime.loop_stages', 'ToolUseStage': 'arc_harness.runtime.loop_stages', 'default_loop_stages': 'arc_harness.runtime.loop_stages', 'delegating_planner_loop_stages': 'arc_harness.runtime.loop_stages', 'DefaultRecoveryPolicy': 'arc_harness.runtime.recovery', 'NoRecoveryPolicy': 'arc_harness.runtime.recovery', 'RecoveryDecision': 'arc_harness.runtime.recovery', 'RecoveryKind': 'arc_harness.runtime.recovery', 'RecoveryPolicy': 'arc_harness.runtime.recovery', 'ArcThread': 'arc_harness.runtime.thread', 'RegisteredTool': 'arc_harness.runtime.tools', 'ToolCall': 'arc_harness.runtime.tools', 'ToolContext': 'arc_harness.runtime.tools', 'ToolDispatcher': 'arc_harness.runtime.tools', 'ToolError': 'arc_harness.runtime.tools', 'ToolRegistry': 'arc_harness.runtime.tools', 'ToolResult': 'arc_harness.runtime.tools', 'ToolSpec': 'arc_harness.runtime.tools', 'default_tool_registry': 'arc_harness.runtime.tools'}

__all__ = sorted(_EXPORTS)

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
