"""Composable stages for the ARC episode loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .actions import Action, Frame, StepRecord
from .agent import ArcAgent
from .checkpoint import CheckpointStore
from .config import RunnerConfig
from .environment import ArcEnvironment, EnvironmentResult
from .events import AgentEvent
from .guardrails import GuardrailDecision
from .hooks import HookManager
from .memory import MemoryManager
from .policy import Decision, HookDecision
from .tracing import Trace
from .validation import validate_action, validate_frame


@dataclass
class LoopState:
    """Mutable state passed through one episode pipeline."""

    episode_id: str
    step: int
    frame: Frame
    total_reward: float = 0.0
    proposed_action: Action | None = None
    decision: HookDecision | None = None
    result: EnvironmentResult | None = None
    record: StepRecord | None = None
    status: str = "RUNNING"
    done: bool = False
    stop: bool = False
    phase: str = "step"
    plan: dict[str, Any] | None = None
    context: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> list[Frame]:
        return self.metadata.get("frames", [])

    def stop_with(self, status: str, *, done: bool = False, reason: str = "") -> None:
        self.status = status
        self.done = done
        self.stop = True
        if reason:
            self.metadata["stop_reason"] = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "status": self.status,
            "done": self.done,
            "stop": self.stop,
            "phase": self.phase,
            "total_reward": self.total_reward,
            "proposed_action": self.proposed_action.to_dict() if self.proposed_action else None,
            "result": _result_to_dict(self.result),
            "metadata": self.metadata,
        }


@dataclass
class LoopRuntime:
    """Runtime services available to loop stages."""

    env: ArcEnvironment
    agent: ArcAgent
    memory: MemoryManager
    hooks: HookManager
    guardrails: list
    config: RunnerConfig
    trace: Trace
    root_span: Any
    checkpoints: CheckpointStore | None = None
    events: list[AgentEvent] = field(default_factory=list)

    def emit(self, event_type: str, episode_id: str, payload: dict[str, Any] | None = None) -> None:
        event = AgentEvent(event_type, episode_id, payload or {})
        self.hooks.emit(event)
        self.events.append(event)

    def drain_events(self) -> list[AgentEvent]:
        events = list(self.events)
        self.events.clear()
        return events

    def checkpoint(self, episode_id: str, step: int, frame: Frame) -> None:
        if self.config.checkpoint and self.checkpoints is not None:
            self.checkpoints.write(episode_id, step, frame, self.memory.working.steps, extra={"config": self.config.to_dict()})

    def run_action_guardrails(self, step: int, frame: Frame, action: Action) -> Action:
        next_action = action
        for guardrail in self.guardrails:
            method = getattr(guardrail, "check_action", None)
            if not method:
                continue
            result = method(step, frame, next_action)
            if result.decision == GuardrailDecision.FAIL:
                raise RuntimeError(f"Action guardrail failed: {result.reason}")
            if result.decision == GuardrailDecision.REWRITE and result.action is not None:
                next_action = result.action
        return next_action

    def run_frame_guardrails(self, frame: Frame, phase: str) -> None:
        for guardrail in self.guardrails:
            method = getattr(guardrail, "check_frame", None)
            if not method:
                continue
            result = method(frame, phase)
            if result.decision == GuardrailDecision.FAIL:
                raise RuntimeError(f"Frame guardrail failed: {result.reason}")

    def run_result_guardrails(self, step: int, result: EnvironmentResult) -> None:
        for guardrail in self.guardrails:
            method = getattr(guardrail, "check_result", None)
            if not method:
                continue
            outcome = method(step, result)
            if outcome.decision == GuardrailDecision.FAIL:
                raise RuntimeError(f"Result guardrail failed: {outcome.reason}")


class LoopStage(Protocol):
    name: str

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        ...


class StagePipeline:
    """Ordered, inspectable loop stage pipeline."""

    def __init__(self, stages: list[LoopStage] | None = None, *, emit_stage_events: bool = True) -> None:
        self.stages = list(stages or default_loop_stages())
        self.emit_stage_events = emit_stage_events

    def insert_before(self, target: str, stage: LoopStage) -> None:
        self.stages.insert(self._index(target), stage)

    def insert_after(self, target: str, stage: LoopStage) -> None:
        self.stages.insert(self._index(target) + 1, stage)

    def replace(self, target: str, stage: LoopStage) -> None:
        self.stages[self._index(target)] = stage

    def names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    def run_step(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        for stage in self.stages:
            if state.stop:
                break
            if self.emit_stage_events:
                runtime.emit("stage.started", state.episode_id, {"step": state.step, "stage": stage.name})
            try:
                state.phase = stage.name
                state = stage.run(state, runtime)
            except Exception as exc:
                if self.emit_stage_events:
                    runtime.emit("stage.failed", state.episode_id, {"step": state.step, "stage": stage.name, "error": repr(exc)})
                raise
            if self.emit_stage_events:
                runtime.emit(
                    "stage.completed",
                    state.episode_id,
                    {"step": state.step, "stage": stage.name, "status": state.status, "stop": state.stop},
                )
        return state

    def _index(self, target: str) -> int:
        for idx, stage in enumerate(self.stages):
            if stage.name == target:
                return idx
        raise ValueError(f"Loop stage {target!r} was not found.")


class DoneCheckStage:
    name = "done_check"

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        frames = runtime.memory.working.frames
        if runtime.agent.is_done(frames, state.frame, runtime.memory):
            state.stop_with(state.frame.status, done=True, reason="agent_done")
        return state


class BuildContextStage:
    """Optional stage for injecting externally supplied context builders."""

    name = "context.build"

    def __init__(self, builder: Callable[[LoopState, LoopRuntime], Any] | None = None) -> None:
        self.builder = builder

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        if self.builder is not None:
            state.context = self.builder(state, runtime)
            runtime.emit("context.built", state.episode_id, {"step": state.step})
        return state


class DecisionStage:
    name = "decision"

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        action_span = runtime.trace.start_span("agent.choose_action", parent_id=runtime.root_span.span_id, metadata={"step": state.step})
        action = runtime.agent.choose_action(runtime.memory.working.frames, state.frame, runtime.memory)
        action_span.finish({"action": action.to_dict() if hasattr(action, "to_dict") else repr(action)})
        if runtime.config.validate_actions:
            validate_action(action, state.frame)
        action = runtime.run_action_guardrails(state.step, state.frame, action)
        state.proposed_action = action
        runtime.emit("action.proposed", state.episode_id, {"step": state.step, "action": action.to_dict()})
        return state


class PermissionStage:
    name = "permission"

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        if state.proposed_action is None:
            raise RuntimeError("PermissionStage requires a proposed action.")
        decision = runtime.hooks.before_action(state.step, state.frame, state.proposed_action)
        state.decision = decision
        if decision.decision in {Decision.BLOCK, Decision.ASK}:
            runtime.memory.record_failure(f"Action blocked at step {state.step}: {decision.reason}", durable=False)
            event_name = "action.permission_requested" if decision.decision == Decision.ASK else "action.blocked"
            status = "WAITING_FOR_APPROVAL" if decision.decision == Decision.ASK else "BLOCKED"
            runtime.emit(event_name, state.episode_id, {"step": state.step, "reason": decision.reason})
            state.stop_with(status, done=False, reason=decision.reason)
            return state

        action = decision.action or state.proposed_action
        if runtime.config.validate_actions:
            validate_action(action, state.frame)
        action = runtime.run_action_guardrails(state.step, state.frame, action)
        state.proposed_action = action
        if decision.decision == Decision.REWRITE:
            runtime.emit("action.rewritten", state.episode_id, {"step": state.step, "action": action.to_dict(), "reason": decision.reason})
        return state


class ActionExecutionStage:
    name = "action.execute"

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        if state.proposed_action is None:
            raise RuntimeError("ActionExecutionStage requires a proposed action.")
        env_span = runtime.trace.start_span("env.step", parent_id=runtime.root_span.span_id, metadata={"step": state.step, "action": state.proposed_action.to_dict()})
        result = runtime.env.step(state.proposed_action)
        env_span.finish({"status": result.frame.status, "reward": result.reward, "done": result.done})
        if runtime.config.validate_frames:
            validate_frame(result.frame)
        runtime.run_result_guardrails(state.step, result)

        record = StepRecord(step=state.step, before=state.frame, action=state.proposed_action, after=result.frame, reward=result.reward, info=result.info)
        runtime.memory.working.remember_step(record)
        runtime.checkpoint(state.episode_id, state.step, result.frame)
        runtime.hooks.after_action(record)
        runtime.emit("action.completed", state.episode_id, {"record": record.to_dict(include_grids=False)})

        state.result = result
        state.record = record
        state.frame = result.frame
        state.total_reward += result.reward
        return state


class StopCheckStage:
    name = "stop_check"

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        if state.result is not None and state.result.done:
            state.stop_with(state.frame.status, done=True, reason="environment_done")
            return state
        if runtime.agent.is_done(runtime.memory.working.frames, state.frame, runtime.memory):
            state.stop_with(state.frame.status, done=True, reason="agent_done_after_action")
            return state
        if runtime.memory.working.detects_loop(window=runtime.config.loop_window):
            runtime.memory.record_failure(f"Loop detected in episode {state.episode_id} at step {state.step}.", durable=False)
            runtime.emit("loop.detected", state.episode_id, {"step": state.step})
            if runtime.config.stop_on_loop:
                state.stop_with("LOOP_DETECTED", done=False, reason="loop_detected")
        return state


def default_loop_stages() -> list[LoopStage]:
    return [
        DoneCheckStage(),
        DecisionStage(),
        PermissionStage(),
        ActionExecutionStage(),
        StopCheckStage(),
    ]


def _result_to_dict(result: EnvironmentResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "frame": result.frame.to_dict(include_grid=False),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }
