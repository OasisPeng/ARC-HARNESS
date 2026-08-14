"""Tool calling primitives for ARC agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from arc_harness.core.actions import Action, ActionType, Frame
from arc_harness.models.delegation import DelegationManager
from arc_harness.memory.memory import MemoryManager


class ToolError(RuntimeError):
    """Raised when tool registration or dispatch fails."""


@dataclass(frozen=True)
class ToolSpec:
    """Stable metadata and lightweight schema for one tool."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": list(self.required),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ToolCall:
    """A model/agent request to execute one named tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: f"tool_{uuid.uuid4().hex}")
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "ToolCall | dict[str, Any]") -> "ToolCall":
        if isinstance(value, ToolCall):
            return value
        if isinstance(value, dict):
            return cls(
                name=str(value["name"]),
                arguments=dict(value.get("arguments", {})),
                call_id=str(value.get("call_id") or f"tool_{uuid.uuid4().hex}"),
                metadata=dict(value.get("metadata", {})),
            )
        raise TypeError(f"Cannot convert {value!r} to ToolCall.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ToolResult:
    """Structured result returned by a tool dispatch."""

    call_id: str
    name: str
    ok: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "ok": self.ok,
            "output": _jsonify(self.output),
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ToolContext:
    """Runtime state exposed to tool implementations."""

    memory: MemoryManager
    frame: Frame | None = None
    frames: list[Frame] = field(default_factory=list)
    delegation: DelegationManager | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    spec: ToolSpec

    def __call__(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        ...


ToolFunction = Callable[[dict[str, Any], ToolContext], Any]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: ToolFunction

    def __call__(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        return self.handler(arguments, context)


class ToolRegistry:
    """Register and resolve executable tools by name."""

    def __init__(self, tools: Iterable[Tool | RegisteredTool] | None = None) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(
        self,
        tool: Tool | RegisteredTool | ToolFunction,
        spec: ToolSpec | None = None,
    ) -> RegisteredTool:
        if isinstance(tool, RegisteredTool):
            registered = tool
        else:
            resolved_spec = spec or getattr(tool, "spec", None)
            if not isinstance(resolved_spec, ToolSpec):
                raise ToolError("Tool registration requires a ToolSpec.")
            registered = RegisteredTool(resolved_spec, tool)  # type: ignore[arg-type]
        if registered.spec.name in self._tools:
            raise ToolError(f"Tool {registered.spec.name!r} is already registered.")
        self._tools[registered.spec.name] = registered
        return registered

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        tool = self.get(name)
        if tool is None:
            raise ToolError(f"Tool {name!r} is not registered.")
        return tool

    def list(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def to_dict(self) -> dict[str, Any]:
        return {"tools": [spec.to_dict() for spec in self.list()]}


class ToolDispatcher:
    """Validate, authorize, execute, and normalize tool calls."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        permissions: dict[str, str] | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self.registry = registry or default_tool_registry()
        self.permissions = dict(permissions or {})
        self.raise_on_error = raise_on_error

    @classmethod
    def with_default_tools(cls, **kwargs) -> "ToolDispatcher":
        return cls(default_tool_registry(), **kwargs)

    def dispatch(self, call: ToolCall | dict[str, Any], context: ToolContext) -> ToolResult:
        tool_call = ToolCall.from_value(call)
        try:
            tool = self.registry.require(tool_call.name)
            self._check_permission(tool_call.name)
            self._validate_arguments(tool.spec, tool_call.arguments)
            output = tool(tool_call.arguments, context)
            return ToolResult(tool_call.call_id, tool_call.name, True, output=output)
        except Exception as exc:
            if self.raise_on_error:
                raise
            return ToolResult(tool_call.call_id, tool_call.name, False, error=str(exc), metadata={"error_type": type(exc).__name__})

    def dispatch_many(self, calls: Iterable[ToolCall | dict[str, Any]], context: ToolContext) -> list[ToolResult]:
        return [self.dispatch(call, context) for call in calls]

    def _check_permission(self, name: str) -> None:
        decision = self.permissions.get(name, "allow")
        if decision == "deny":
            raise ToolError(f"Tool {name!r} is denied by policy.")
        if decision == "ask":
            raise ToolError(f"Tool {name!r} requires human approval.")
        if decision != "allow":
            raise ToolError(f"Unknown tool permission {decision!r} for {name!r}.")

    def _validate_arguments(self, spec: ToolSpec, arguments: dict[str, Any]) -> None:
        missing = [name for name in spec.required if name not in arguments]
        if missing:
            raise ToolError(f"Tool {spec.name!r} missing required argument(s): {', '.join(missing)}.")
        properties = spec.parameters.get("properties", {}) if isinstance(spec.parameters, dict) else {}
        for name, schema in properties.items():
            if name not in arguments:
                continue
            expected = schema.get("type") if isinstance(schema, dict) else None
            if expected and not _matches_json_type(arguments[name], expected):
                raise ToolError(f"Tool {spec.name!r} argument {name!r} must be {expected}.")


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_observe_objects, _tool_spec("observe_objects", "Summarize connected objects in the current frame."))
    registry.register(_propose_actions, _tool_spec("propose_actions", "Propose untried ARC environment actions.", {"limit": "integer"}))
    registry.register(_search_memory, _tool_spec("search_memory", "Search durable memory.", {"query": "string", "limit": "integer"}, required=("query",)))
    registry.register(_write_note, _tool_spec("write_note", "Write a note to working memory.", {"text": "string"}, required=("text",)))
    registry.register(_delegate, _tool_spec("delegate", "Delegate a task to a registered subagent.", {"kind": "string", "payload": "object", "budget": "integer"}, required=("kind",)))
    return registry


def _tool_spec(name: str, description: str, properties: dict[str, str] | None = None, required: tuple[str, ...] = ()) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {key: {"type": value} for key, value in dict(properties or {}).items()},
        },
        required=required,
    )


def _observe_objects(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    frame = _frame(context)
    components = _connected_components(frame)
    return {
        "status": frame.status,
        "width": frame.width,
        "height": frame.height,
        "object_count": len(components),
        "nonzero_object_count": len([item for item in components if item["color"] != 0]),
        "objects": components[: int(arguments.get("limit", 12))],
    }


def _propose_actions(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    frame = _frame(context)
    limit = int(arguments.get("limit", 16))
    tried = {_action_key(action) for action in context.memory.working.recent_actions(limit=frame.width * frame.height + 16)}
    actions: list[Action] = []
    for y in range(frame.height):
        for x in range(frame.width):
            action = Action(ActionType.ACTION6, (x, y))
            if _action_key(action) not in tried:
                actions.append(action)
            if len(actions) >= limit:
                return {"actions": [action.to_dict() for action in actions], "competition_values": [action.to_competition_value() for action in actions]}
    for kind in (ActionType.ACTION1, ActionType.ACTION2, ActionType.ACTION3, ActionType.ACTION4, ActionType.ACTION5):
        action = Action(kind)
        if _action_key(action) not in tried:
            actions.append(action)
        if len(actions) >= limit:
            break
    return {"actions": [action.to_dict() for action in actions], "competition_values": [action.to_competition_value() for action in actions]}


def _search_memory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    hits = context.memory.search_entries(str(arguments["query"]), limit=int(arguments.get("limit", 5)), mode="hybrid")
    return {
        "hits": [
            {
                "text": hit.text,
                "score": hit.score,
                "category": hit.category,
                "namespace": list(hit.namespace),
                "tags": list(hit.tags),
                "metadata": hit.metadata,
            }
            for hit in hits
        ]
    }


def _write_note(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    text = str(arguments["text"])
    context.memory.add_note(text)
    return {"written": True, "text": text}


def _delegate(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if context.delegation is None:
        raise ToolError("delegate tool requires a DelegationManager.")
    result = context.delegation.delegate(
        str(arguments["kind"]),
        dict(arguments.get("payload", {})),
        context.memory,
        budget=int(arguments.get("budget", 1000)),
        metadata=dict(arguments.get("metadata", {})),
    )
    return result.to_dict()


def _frame(context: ToolContext) -> Frame:
    if context.frame is None:
        raise ToolError("Tool requires a current frame.")
    return context.frame


def _connected_components(frame: Frame) -> list[dict[str, Any]]:
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y, row in enumerate(frame.grid):
        for x, color in enumerate(row):
            if (x, y) in visited:
                continue
            cells = _flood_fill(frame, x, y, color, visited)
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            bbox = {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": bbox,
                    "center": ((bbox["x1"] + bbox["x2"]) // 2, (bbox["y1"] + bbox["y2"]) // 2),
                    "sample_cells": cells[:8],
                }
            )
    components.sort(key=lambda item: (item["color"] == 0, -item["size"], item["color"], item["bbox"]["y1"], item["bbox"]["x1"]))
    return components


def _flood_fill(frame: Frame, start_x: int, start_y: int, color: int, visited: set[tuple[int, int]]) -> list[tuple[int, int]]:
    stack = [(start_x, start_y)]
    visited.add((start_x, start_y))
    cells: list[tuple[int, int]] = []
    while stack:
        x, y = stack.pop()
        cells.append((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or ny >= frame.height or nx >= frame.width:
                continue
            if (nx, ny) in visited or frame.grid[ny][nx] != color:
                continue
            visited.add((nx, ny))
            stack.append((nx, ny))
    return cells


def _action_key(action: Action) -> tuple[str, tuple[int, int] | None]:
    kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
    return kind, action.xy


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _jsonify(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value
