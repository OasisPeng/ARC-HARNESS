"""Lazy exports for the memory package."""

from importlib import import_module
from typing import Any

_EXPORTS = {'ContextBudget': 'arc_harness.memory.context', 'ContextBundle': 'arc_harness.memory.context', 'ContextInjector': 'arc_harness.memory.context', 'ContextManager': 'arc_harness.memory.context', 'ContextRole': 'arc_harness.memory.context', 'ContextSection': 'arc_harness.memory.context', 'DurableMemory': 'arc_harness.memory.memory', 'MemoryEntry': 'arc_harness.memory.memory', 'MemoryManager': 'arc_harness.memory.memory', 'WorkingMemory': 'arc_harness.memory.memory', 'MemoryPolicy': 'arc_harness.memory.memory_policy', 'LightweightEmbeddingIndex': 'arc_harness.memory.memory_store', 'SearchResult': 'arc_harness.memory.memory_store', 'StructuredMemoryStore': 'arc_harness.memory.memory_store'}

__all__ = sorted(_EXPORTS)

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
