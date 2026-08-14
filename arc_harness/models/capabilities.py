"""Capability registry for pluggable harness runtime providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


class CapabilityError(RuntimeError):
    """Raised when a required runtime capability is missing or invalid."""


@dataclass(frozen=True)
class ProviderDescriptor:
    """Stable metadata for one registered capability provider."""

    capability: str
    name: str
    version: str = "0.1"
    supports: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability:
            raise ValueError("capability must be non-empty.")
        if not self.name:
            raise ValueError("name must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "name": self.name,
            "version": self.version,
            "supports": list(self.supports),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CapabilityRegistration:
    """A provider and its descriptor as stored in the registry."""

    descriptor: ProviderDescriptor
    provider: Any

    def to_dict(self) -> dict[str, Any]:
        return self.descriptor.to_dict()


class DescribedProvider(Protocol):
    descriptor: ProviderDescriptor


class CapabilityRegistry:
    """Register and resolve runtime providers by capability and name.

    The registry is intentionally small: it gives the harness a common seam for
    models, environments, subagents, sandboxes, evaluators, and future plugins
    without forcing any one provider base class.
    """

    def __init__(self, providers: Iterable[Any] | None = None) -> None:
        self._providers: dict[str, dict[str, CapabilityRegistration]] = {}
        for provider in providers or ():
            self.register(provider)

    def register(self, provider: Any, descriptor: ProviderDescriptor | None = None) -> Any:
        resolved = descriptor or getattr(provider, "descriptor", None)
        if not isinstance(resolved, ProviderDescriptor):
            raise CapabilityError("Provider registration requires a ProviderDescriptor.")
        bucket = self._providers.setdefault(resolved.capability, {})
        if resolved.name in bucket:
            raise CapabilityError(f"Provider {resolved.capability}/{resolved.name} is already registered.")
        bucket[resolved.name] = CapabilityRegistration(resolved, provider)
        return provider

    def unregister(self, capability: str, name: str) -> Any:
        bucket = self._providers.get(capability, {})
        if name not in bucket:
            raise CapabilityError(f"Provider {capability}/{name} is not registered.")
        return bucket.pop(name).provider

    def get(self, capability: str, name: str | None = None) -> Any | None:
        registration = self.get_registration(capability, name=name)
        return registration.provider if registration else None

    def get_registration(self, capability: str, name: str | None = None) -> CapabilityRegistration | None:
        bucket = self._providers.get(capability, {})
        if not bucket:
            return None
        if name is None:
            return next(iter(bucket.values()))
        return bucket.get(name)

    def require(
        self,
        capability: str,
        name: str | None = None,
        *,
        supports: Iterable[str] = (),
    ) -> Any:
        registration = self.get_registration(capability, name=name)
        provider_name = name or "<default>"
        if registration is None:
            raise CapabilityError(f"Missing required provider {capability}/{provider_name}.")
        missing = tuple(feature for feature in supports if feature not in registration.descriptor.supports)
        if missing:
            raise CapabilityError(
                f"Provider {capability}/{registration.descriptor.name} lacks required support: {', '.join(missing)}."
            )
        return registration.provider

    def list(self, capability: str | None = None) -> list[ProviderDescriptor]:
        if capability is not None:
            return [registration.descriptor for registration in self._providers.get(capability, {}).values()]
        descriptors: list[ProviderDescriptor] = []
        for bucket in self._providers.values():
            descriptors.extend(registration.descriptor for registration in bucket.values())
        return descriptors

    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            capability: [registration.to_dict() for registration in bucket.values()]
            for capability, bucket in sorted(self._providers.items())
        }


def make_default_capability_registry() -> CapabilityRegistry:
    """Create a registry with built-in lightweight runtime providers."""

    registry = CapabilityRegistry()
    try:
        from arc_harness.integrations.sandbox import LocalSubprocessSandbox

        registry.register(LocalSubprocessSandbox())
    except Exception:
        # Importing the package should never fail just because an optional
        # built-in provider cannot be constructed in a constrained runtime.
        pass
    return registry


DEFAULT_CAPABILITY_REGISTRY = make_default_capability_registry()
