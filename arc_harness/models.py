"""Offline-friendly model integration for ARC agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .actions import Action, Frame
from .agent import ArcAgent, DelegatingPlannerAgent
from .memory import MemoryManager


@dataclass(frozen=True)
class ModelInput:
    frames: list[Frame]
    latest_frame: Frame
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_dict(include_grid=True) for frame in self.frames],
            "latest_frame": self.latest_frame.to_dict(include_grid=True),
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ModelOutput:
    action: Action | None = None
    plan: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    rationale: str = ""
    raw: Any = None

    def best_action(self) -> Action:
        if self.action is not None:
            return self.action
        if self.plan:
            return Action.from_value(self.plan[0]["action"])
        raise ValueError("ModelOutput does not contain an action or non-empty plan.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict() if self.action else None,
            "plan": self.plan,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw": self.raw,
        }


class LocalModel(Protocol):
    """Protocol for Kaggle-safe local models or deterministic policies."""

    name: str

    def predict(self, model_input: ModelInput) -> ModelOutput:
        ...


class CallableModel:
    """Wrap a Python callable as a local model backend."""

    name = "CallableModel"

    def __init__(self, fn: Callable[[ModelInput], Action | ModelOutput | dict | tuple | str], name: str | None = None) -> None:
        self.fn = fn
        if name:
            self.name = name

    def predict(self, model_input: ModelInput) -> ModelOutput:
        return coerce_model_output(self.fn(model_input))


class JsonPolicyModel:
    """Read simple frame-signature rules from JSON for fully offline inference."""

    name = "JsonPolicyModel"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rules = json.loads(self.path.read_text(encoding="utf-8"))

    def predict(self, model_input: ModelInput) -> ModelOutput:
        signature = frame_signature(model_input.latest_frame)
        value = self.rules.get(signature) or self.rules.get("*")
        if value is None:
            raise ValueError(f"No JSON policy rule for frame signature {signature!r}.")
        return coerce_model_output(value)


class ModelBackedAgent(ArcAgent):
    """Use a local model backend first, then optionally fall back to a planner."""

    def __init__(self, model: LocalModel, *, fallback: ArcAgent | None = None, inject_context: bool = True) -> None:
        self.model = model
        self.fallback = fallback or DelegatingPlannerAgent()
        self.inject_context = inject_context

    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        context = memory.durable.search("action effect rule procedure", limit=5) if self.inject_context else []
        model_input = ModelInput(frames, latest_frame, context="\n\n".join(context), metadata={"model": self.model.name})
        try:
            output = self.model.predict(model_input)
            action = output.best_action()
            memory.add_note(f"{self.model.name} selected {action.to_competition_value()}: {output.rationale}")
            memory.add_fact(
                f"{self.model.name} prediction: {action.to_competition_value()} confidence={output.confidence}",
                category="model",
                namespace=("models", self.model.name),
                tags=("model", self.model.name),
                confidence=output.confidence,
                metadata=output.to_dict(),
            )
            return action
        except Exception as exc:
            memory.record_failure(f"{self.model.name} failed, falling back: {exc}", durable=False)
            return self.fallback.choose_action(frames, latest_frame, memory)


class ModelRegistry:
    """Small registry for local model factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., LocalModel]] = {}

    def register(self, name: str, factory: Callable[..., LocalModel]) -> None:
        if not name:
            raise ValueError("Model name must be non-empty.")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> LocalModel:
        if name not in self._factories:
            raise KeyError(f"Unknown local model {name!r}. Registered: {sorted(self._factories)}")
        return self._factories[name](**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


DEFAULT_MODEL_REGISTRY = ModelRegistry()
DEFAULT_MODEL_REGISTRY.register("json_policy", JsonPolicyModel)


def load_model_from_config(path: str | Path, registry: ModelRegistry | None = None) -> LocalModel:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    name = str(config.get("type") or config.get("name"))
    kwargs = dict(config.get("kwargs", {}))
    if "path" in config and "path" not in kwargs:
        kwargs["path"] = config["path"]
    return (registry or DEFAULT_MODEL_REGISTRY).create(name, **kwargs)


def build_agent_from_model_config(path: str | Path, *, fallback: ArcAgent | None = None) -> ModelBackedAgent:
    return ModelBackedAgent(load_model_from_config(path), fallback=fallback)


def coerce_model_output(value: Action | ModelOutput | dict | tuple | str) -> ModelOutput:
    if isinstance(value, ModelOutput):
        return value
    if isinstance(value, Action):
        return ModelOutput(action=value)
    if isinstance(value, dict):
        if "plan" in value:
            plan = list(value.get("plan") or [])
            action = Action.from_value(value["action"]) if value.get("action") is not None else None
            return ModelOutput(action=action, plan=plan, confidence=float(value.get("confidence", 1.0)), rationale=str(value.get("rationale", "")), raw=value)
        return ModelOutput(action=Action.from_value(value), raw=value)
    return ModelOutput(action=Action.from_value(value), raw=value)


def frame_signature(frame: Frame) -> str:
    return "|".join(",".join(str(cell) for cell in row) for row in frame.grid)
