"""Lightweight ARC-AGI-3-style agent harness."""

__version__ = "0.1.0"

from .actions import Action, ActionType, Frame, StepRecord
from .agent import ArcAgent, DelegatingPlannerAgent, HandoffAgent, HeuristicAgent, RuleLearningAgent
from .checkpoint import CheckpointStore
from .config import RunnerConfig
from .context import ContextBudget, ContextBundle, ContextInjector, ContextManager, ContextRole, ContextSection
from .delegation import (
    DelegationConfig,
    DelegationError,
    DelegationManager,
    HandoffController,
    HandoffRecord,
    HandoffRule,
    SubAgent,
    SubAgentResult,
    SubTask,
)
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
from .models import (
    DEFAULT_MODEL_REGISTRY,
    CallableModel,
    JsonPolicyModel,
    LocalModel,
    ModelBackedAgent,
    ModelInput,
    ModelOutput,
    ModelRegistry,
    build_agent_from_model_config,
    load_model_from_config,
)
from .official import (
    ArcAgi3Config,
    EnvironmentFileCatalog,
    OfficialArcEnvironment,
    OfficialDependencyError,
    coerce_official_frame,
    create_official_environment,
    resolve_official_action,
)
from .official_eval import OfficialSmokeReport, OfficialSmokeResult, OfficialSmokeRunner, discover_official_games
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
    "ArcAgi3Config",
    "ArcAgent",
    "ArcEnvironment",
    "ArcThread",
    "CheckpointStore",
    "CallableModel",
    "CoordinateBoundsGuardrail",
    "ContextBudget",
    "ContextBundle",
    "ContextInjector",
    "ContextManager",
    "ContextRole",
    "ContextSection",
    "DEFAULT_MODEL_REGISTRY",
    "DurableMemory",
    "AgentEvent",
    "Decision",
    "DelegatingPlannerAgent",
    "DelegationConfig",
    "DelegationError",
    "DelegationManager",
    "DiffSubAgent",
    "EnvironmentResult",
    "EnvironmentFileCatalog",
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
    "HandoffAgent",
    "HandoffController",
    "HandoffRecord",
    "HandoffRule",
    "HeuristicAgent",
    "HookDecision",
    "Hook",
    "HookMatcher",
    "HookManager",
    "JsonlTraceHook",
    "MemoryEntry",
    "MemoryManager",
    "MemoryPolicy",
    "ModelBackedAgent",
    "ModelInput",
    "ModelOutput",
    "ModelRegistry",
    "LocalModel",
    "JsonPolicyModel",
    "OfficialArcEnvironment",
    "OfficialDependencyError",
    "OfficialSmokeReport",
    "OfficialSmokeResult",
    "OfficialSmokeRunner",
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
    "build_agent_from_model_config",
    "coerce_official_frame",
    "create_official_environment",
    "discover_official_games",
    "load_model_from_config",
    "resolve_official_action",
    "validate_action",
    "validate_environment",
    "validate_frame",
]
