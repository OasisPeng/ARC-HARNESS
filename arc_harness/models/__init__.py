"""Lazy exports for the models package."""

from importlib import import_module
from typing import Any

_EXPORTS = {'DEFAULT_CAPABILITY_REGISTRY': 'arc_harness.models.capabilities', 'CapabilityError': 'arc_harness.models.capabilities', 'CapabilityRegistration': 'arc_harness.models.capabilities', 'CapabilityRegistry': 'arc_harness.models.capabilities', 'ProviderDescriptor': 'arc_harness.models.capabilities', 'make_default_capability_registry': 'arc_harness.models.capabilities', 'DelegationConfig': 'arc_harness.models.delegation', 'DelegationError': 'arc_harness.models.delegation', 'DelegationManager': 'arc_harness.models.delegation', 'HandoffController': 'arc_harness.models.delegation', 'HandoffRecord': 'arc_harness.models.delegation', 'HandoffRule': 'arc_harness.models.delegation', 'SubAgent': 'arc_harness.models.delegation', 'SubAgentResult': 'arc_harness.models.delegation', 'SubTask': 'arc_harness.models.delegation', 'DEFAULT_MODEL_REGISTRY': 'arc_harness.models.models', 'CallableModel': 'arc_harness.models.models', 'CandidateAction': 'arc_harness.models.models', 'CandidateGenerator': 'arc_harness.models.models', 'CandidateRankingAgent': 'arc_harness.models.models', 'JsonPolicyModel': 'arc_harness.models.models', 'LocalModel': 'arc_harness.models.models', 'ModelBackedAgent': 'arc_harness.models.models', 'ModelInput': 'arc_harness.models.models', 'ModelOutput': 'arc_harness.models.models', 'ModelRegistry': 'arc_harness.models.models', 'QwenLocalRanker': 'arc_harness.models.models', 'build_agent_from_model_config': 'arc_harness.models.models', 'load_model_from_config': 'arc_harness.models.models', 'DiffSubAgent': 'arc_harness.models.subagents', 'ExplorerSubAgent': 'arc_harness.models.subagents', 'PerceptionSubAgent': 'arc_harness.models.subagents', 'PlannerSubAgent': 'arc_harness.models.subagents'}

__all__ = sorted(_EXPORTS)

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
