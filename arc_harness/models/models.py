"""Offline-friendly model integration for ARC agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from arc_harness.core.actions import Action, ActionType, Frame
from arc_harness.core.agent import ArcAgent, DelegatingPlannerAgent
from arc_harness.memory.context import ContextManager
from arc_harness.memory.memory import MemoryManager


@dataclass(frozen=True)
class ModelInput:
    frames: list[Frame]
    latest_frame: Frame
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    candidates: list["CandidateAction"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_dict(include_grid=True) for frame in self.frames],
            "latest_frame": self.latest_frame.to_dict(include_grid=True),
            "context": self.context,
            "metadata": self.metadata,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class CandidateAction:
    """A bounded action choice the local model can rank."""

    candidate_id: int
    action: Action
    reason: str = ""
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action": self.action.to_dict(),
            "competition_value": self.action.to_competition_value(),
            "reason": self.reason,
            "features": self.features,
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


class CandidateGenerator:
    """Generate a compact offline action set for ranking models."""

    def __init__(self, *, max_coordinate_candidates: int = 64, include_simple_actions: bool = True) -> None:
        self.max_coordinate_candidates = max_coordinate_candidates
        self.include_simple_actions = include_simple_actions

    def generate(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> list[CandidateAction]:
        tried = {_action_key(action) for action in memory.working.recent_actions(limit=latest_frame.width * latest_frame.height + 32)}
        previous = frames[-2] if len(frames) >= 2 else None
        candidates: list[CandidateAction] = []

        for action, reason, features in self._coordinate_actions(latest_frame, previous):
            if _action_key(action) in tried:
                continue
            candidates.append(CandidateAction(len(candidates), action, reason, features))
            if len([candidate for candidate in candidates if candidate.action.xy is not None]) >= self.max_coordinate_candidates:
                break

        if self.include_simple_actions:
            for kind in [
                ActionType.ACTION1,
                ActionType.ACTION2,
                ActionType.ACTION3,
                ActionType.ACTION4,
                ActionType.ACTION5,
                ActionType.ACTION7,
            ]:
                action = Action(kind)
                if _action_key(action) not in tried:
                    candidates.append(CandidateAction(len(candidates), action, "try simple action", {"kind": kind.value}))

        if not candidates:
            candidates.append(CandidateAction(0, Action(ActionType.ACTION1), "fallback when all generated actions were tried"))
        return candidates

    def _coordinate_actions(self, frame: Frame, previous: Frame | None) -> list[tuple[Action, str, dict[str, Any]]]:
        changed = self._changed_cells(frame, previous)
        interesting = changed or self._interesting_cells(frame)
        if not interesting and frame.width and frame.height:
            interesting = [(x, y, frame.grid[y][x], "scan") for y in range(frame.height) for x in range(frame.width)]
        return [
            (
                Action(ActionType.ACTION6, (x, y)),
                f"probe {source} cell value={value}",
                {"x": x, "y": y, "value": value, "source": source},
            )
            for x, y, value, source in interesting
        ]

    def _interesting_cells(self, frame: Frame) -> list[tuple[int, int, int, str]]:
        cells: list[tuple[int, int, int, str]] = []
        seen_values: set[int] = set()
        for y, row in enumerate(frame.grid):
            for x, value in enumerate(row):
                if value != 0 and value not in seen_values:
                    cells.append((x, y, value, "first-nonzero"))
                    seen_values.add(value)
        if frame.width and frame.height:
            for x, y in [(0, 0), (frame.width - 1, 0), (0, frame.height - 1), (frame.width - 1, frame.height - 1)]:
                cells.append((x, y, frame.grid[y][x], "corner"))
        return _dedupe_cells(cells)

    def _changed_cells(self, frame: Frame, previous: Frame | None) -> list[tuple[int, int, int, str]]:
        if previous is None:
            return []
        cells: list[tuple[int, int, int, str]] = []
        for y, row in enumerate(frame.grid):
            for x, value in enumerate(row):
                old = previous.grid[y][x] if y < previous.height and x < previous.width else None
                if old != value:
                    cells.append((x, y, value, "changed"))
        return _dedupe_cells(cells)


class QwenLocalRanker:
    """Rank candidate ARC actions with an offline Qwen causal language model.

    The model path must already exist locally, for example under
    `/kaggle/input/qwen2-5-0-5b-instruct`. No network calls are made.
    """

    name = "Qwen2.5-0.5B-Instruct"

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 128,
        device_map: str | None = "auto",
        torch_dtype: str | None = "auto",
        trust_remote_code: bool = False,
        load_on_init: bool = True,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.tokenizer = tokenizer
        self.model = model
        if load_on_init and (self.tokenizer is None or self.model is None):
            self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {self.model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("QwenLocalRanker requires transformers to be installed in the offline Kaggle image or package.") from exc

        load_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True, **load_kwargs)

    def predict(self, model_input: ModelInput) -> ModelOutput:
        if self.tokenizer is None or self.model is None:
            self.load()
        if not model_input.candidates:
            raise ValueError("QwenLocalRanker requires ModelInput.candidates.")
        prompt = self._build_prompt(model_input)
        response = self._generate_text(prompt)
        return self._parse_response(response, model_input.candidates)

    def _build_prompt(self, model_input: ModelInput) -> str:
        candidates = [
            {
                "candidate_id": candidate.candidate_id,
                "action": candidate.action.to_dict(),
                "reason": candidate.reason,
                "features": candidate.features,
            }
            for candidate in model_input.candidates
        ]
        payload = {
            "status": model_input.latest_frame.status,
            "grid": model_input.latest_frame.grid,
            "context": model_input.context[-4000:],
            "candidates": candidates,
        }
        return (
            "You rank candidate actions for an ARC-AGI-3 game. "
            "Pick actions that maximize information gain, avoid repeated failures, and progress toward WIN. "
            "Return only one compact JSON object for the best action: "
            "{\"candidate_id\":0,\"score\":0.9,\"reason\":\"...\"}. "
            "Do not return a list, markdown, or extra prose.\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def _generate_text(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an offline ARC action ranking model. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        tokenizer = self.tokenizer
        model = self.model
        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer([messages[-1]["content"]], return_tensors="pt")
        if hasattr(model, "device") and hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_len:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def _parse_response(self, response: str, candidates: list[CandidateAction]) -> ModelOutput:
        parsed = _extract_json(response)
        ranked = _ranked_candidate_rows(parsed)
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        plan: list[dict[str, Any]] = []
        used: set[int] = set()
        for index, row in enumerate(ranked):
            candidate_id = int(row.get("candidate_id", -1))
            candidate = by_id.get(candidate_id)
            if candidate is None or candidate_id in used:
                continue
            used.add(candidate_id)
            plan.append(
                {
                    "candidate_id": candidate_id,
                    "action": candidate.action.to_dict(),
                    "score": float(row.get("score", 1.0 - index * 0.05)),
                    "reason": str(row.get("reason") or candidate.reason),
                }
            )
        for candidate in candidates:
            if candidate.candidate_id not in used:
                plan.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "action": candidate.action.to_dict(),
                        "score": 0.0,
                        "reason": candidate.reason,
                    }
                )
        confidence = float(plan[0]["score"]) if plan else 0.0
        return ModelOutput(plan=plan, confidence=confidence, rationale=plan[0]["reason"] if plan else "", raw={"text": response, "parsed": parsed})


class CandidateRankingAgent(ArcAgent):
    """Agent that lets a local model rank bounded candidate actions."""

    def __init__(
        self,
        model: LocalModel,
        *,
        generator: CandidateGenerator | None = None,
        fallback: ArcAgent | None = None,
        inject_context: bool = True,
    ) -> None:
        self.model = model
        self.generator = generator or CandidateGenerator()
        self.fallback = fallback or DelegatingPlannerAgent()
        self.inject_context = inject_context

    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        candidates = self.generator.generate(frames, latest_frame, memory)
        context = (
            ContextManager().build(
                memory=memory,
                latest_frame=latest_frame,
                query="candidate action effect rule procedure recovery plan information gain",
                include_arc_state=True,
            ).render()
            if self.inject_context
            else ""
        )
        model_input = ModelInput(
            frames,
            latest_frame,
            context=context,
            metadata={"model": self.model.name, "candidate_count": len(candidates)},
            candidates=candidates,
        )
        try:
            output = self.model.predict(model_input)
            action = output.best_action()
            memory.add_note(f"{self.model.name} ranked {len(candidates)} candidates and selected {action.to_competition_value()}: {output.rationale}")
            memory.add_fact(
                f"{self.model.name} ranking selected {action.to_competition_value()} confidence={output.confidence}",
                category="model",
                namespace=("models", self.model.name),
                tags=("model", self.model.name, "ranking"),
                confidence=output.confidence,
                metadata={**output.to_dict(), "candidates": [candidate.to_dict() for candidate in candidates]},
            )
            return action
        except Exception as exc:
            memory.record_failure(f"{self.model.name} ranking failed, falling back: {exc}", durable=False)
            return self.fallback.choose_action(frames, latest_frame, memory)


class ModelBackedAgent(ArcAgent):
    """Use a local model backend first, then optionally fall back to a planner."""

    def __init__(self, model: LocalModel, *, fallback: ArcAgent | None = None, inject_context: bool = True) -> None:
        self.model = model
        self.fallback = fallback or DelegatingPlannerAgent()
        self.inject_context = inject_context

    def choose_action(self, frames: list[Frame], latest_frame: Frame, memory: MemoryManager) -> Action:
        context = (
            ContextManager().build(
                memory=memory,
                latest_frame=latest_frame,
                query="action effect rule procedure recovery plan",
                include_arc_state=True,
            ).render()
            if self.inject_context
            else ""
        )
        model_input = ModelInput(frames, latest_frame, context=context, metadata={"model": self.model.name})
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
DEFAULT_MODEL_REGISTRY.register("qwen2_5_0_5b_instruct", QwenLocalRanker)
DEFAULT_MODEL_REGISTRY.register("qwen_local_ranker", QwenLocalRanker)


def load_model_from_config(path: str | Path, registry: ModelRegistry | None = None) -> LocalModel:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    name = str(config.get("type") or config.get("name"))
    kwargs = dict(config.get("kwargs", {}))
    if "path" in config and "path" not in kwargs:
        kwargs["path"] = config["path"]
    return (registry or DEFAULT_MODEL_REGISTRY).create(name, **kwargs)


def build_agent_from_model_config(path: str | Path, *, fallback: ArcAgent | None = None) -> ArcAgent:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    agent_kind = str(config.get("agent", "model_backed")).strip().lower()
    model = load_model_from_config(path)
    if agent_kind in {"candidate_ranker", "ranking", "ranker"}:
        generator_config = dict(config.get("candidate_generator", {}))
        return CandidateRankingAgent(model, generator=CandidateGenerator(**generator_config), fallback=fallback)
    return ModelBackedAgent(model, fallback=fallback)


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


def _action_key(action: Action) -> tuple[str, tuple[int, int] | None]:
    kind = action.kind.value if isinstance(action.kind, ActionType) else str(action.kind)
    return kind, action.xy


def _dedupe_cells(cells: list[tuple[int, int, int, str]]) -> list[tuple[int, int, int, str]]:
    result: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for x, y, value, source in cells:
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        result.append((x, y, value, source))
    return result


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model returned an empty response.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if match:
        fenced = match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            fallback = _extract_rankings_from_partial_text(fenced)
            if fallback:
                return fallback
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        obj = object_match.group(0)
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            fallback = _extract_rankings_from_partial_text(obj)
            if fallback:
                return fallback
    integer_match = re.search(r"-?\d+", stripped)
    if integer_match:
        return {"candidate_id": int(integer_match.group(0))}
    fallback = _extract_rankings_from_partial_text(stripped)
    if fallback:
        return fallback
    raise ValueError(f"Could not parse model JSON response: {text!r}")


def _ranked_candidate_rows(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [row if isinstance(row, dict) else {"candidate_id": row} for row in parsed]
    if isinstance(parsed, dict):
        if "ranked_candidates" in parsed:
            rows = parsed["ranked_candidates"]
            return [row if isinstance(row, dict) else {"candidate_id": row} for row in rows]
        if "plan" in parsed:
            rows = parsed["plan"]
            return [row if isinstance(row, dict) else {"candidate_id": row} for row in rows]
        if "candidate_id" in parsed:
            return [parsed]
    raise ValueError(f"Model response did not contain candidate rankings: {parsed!r}")


def _extract_rankings_from_partial_text(text: str) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for match in re.finditer(r'"?candidate_id"?\s*:\s*(-?\d+)', text):
        candidate_id = int(match.group(1))
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        window = text[match.start() : match.start() + 240]
        score_match = re.search(r'"?score"?\s*:\s*([0-9.]+)', window)
        score = float(score_match.group(1)) if score_match else max(0.0, 1.0 - len(rows) * 0.05)
        rows.append({"candidate_id": candidate_id, "score": score, "reason": "parsed from partial model response"})
    if rows:
        return {"ranked_candidates": rows}
    return None
