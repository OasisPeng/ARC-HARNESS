"""Lightweight ARC-AGI-3-style agent harness."""

__version__ = "0.1.0"

from .actions import Action, ActionType, Frame, StepRecord
from .agent import ArcAgent, HeuristicAgent, RuleLearningAgent
from .checkpoint import CheckpointStore
from .config import RunnerConfig
from .context import ContextBudget, ContextBundle, ContextInjector, ContextManager, ContextRole, ContextSection
from .delegation import DelegationConfig, DelegationError, DelegationManager, SubAgent, SubAgentResult, SubTask
from .environment import ArcEnvironment, EnvironmentResult, validate_environment
from .errors import ErrorContext, HarnessError, ValidationError
from .evaluation import EvalCase, EvalCaseResult, EvalReport, EvaluationRunner
from .events import AgentEvent
from .guardrails import (
    ActionGuardrail,
    CoordinateBoundsGuardrail,
    FrameGuardrail,
    GuardrailDecision,
    GuardrailResult,
    MaxChangedCellsGuardrail,
    ResultGuardrail,
)
from .hooks import ActionBudgetHook, Hook, HookManager, HookMatcher, JsonlTraceHook
from .loop import EpisodeResult, EpisodeRunner
from .memory import DurableMemory, MemoryEntry, MemoryManager, WorkingMemory
from .memory_policy import MemoryPolicy
from .memory_store import LightweightEmbeddingIndex, SearchResult, StructuredMemoryStore
from .policy import Decision, HookDecision
from .replay import ReplayEpisode, ReplayStep
from .subagents import DiffSubAgent, ExplorerSubAgent, PerceptionSubAgent, PlannerSubAgent
from .thread import ArcThread
from .tracing import Span, Trace, TraceStore
from .validation import validate_action, validate_frame

__all__ = [
    "Action",
    "ActionBudgetHook",
    "ActionGuardrail",
    "ActionType",
    "ArcAgent",
    "ArcEnvironment",
    "ArcThread",
    "CheckpointStore",
    "CoordinateBoundsGuardrail",
    "ContextBudget",
    "ContextBundle",
    "ContextInjector",
    "ContextManager",
    "ContextRole",
    "ContextSection",
    "DurableMemory",
    "AgentEvent",
    "Decision",
    "DelegationConfig",
    "DelegationError",
    "DelegationManager",
    "DiffSubAgent",
    "EnvironmentResult",
    "ErrorContext",
    "EvalCase",
    "EvalCaseResult",
    "EvalReport",
    "EvaluationRunner",
    "ExplorerSubAgent",
    "EpisodeResult",
    "EpisodeRunner",
    "Frame",
    "FrameGuardrail",
    "GuardrailDecision",
    "GuardrailResult",
    "HarnessError",
    "HeuristicAgent",
    "HookDecision",
    "Hook",
    "HookMatcher",
    "HookManager",
    "JsonlTraceHook",
    "MemoryEntry",
    "MemoryManager",
    "MemoryPolicy",
    "PerceptionSubAgent",
    "PlannerSubAgent",
    "ReplayEpisode",
    "ReplayStep",
    "ResultGuardrail",
    "RuleLearningAgent",
    "RunnerConfig",
    "LightweightEmbeddingIndex",
    "SearchResult",
    "MaxChangedCellsGuardrail",
    "Span",
    "StepRecord",
    "SubAgent",
    "SubAgentResult",
    "SubTask",
    "Trace",
    "TraceStore",
    "StructuredMemoryStore",
    "ValidationError",
    "WorkingMemory",
    "validate_action",
    "validate_environment",
    "validate_frame",
]
