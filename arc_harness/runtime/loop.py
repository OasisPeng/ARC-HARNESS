"""Episode runner implementing the core agent loop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Iterator

from arc_harness.core.agent import ArcAgent
from arc_harness.runtime.checkpoint import CheckpointStore
from arc_harness.core.config import RunnerConfig
from arc_harness.core.environment import ArcEnvironment, validate_environment
from arc_harness.utils.errors import ErrorContext
from arc_harness.utils.events import AgentEvent
from arc_harness.runtime.guardrails import GuardrailDecision
from arc_harness.runtime.hooks import HookManager
from arc_harness.models.delegation import DelegationManager
from arc_harness.runtime.loop_stages import LoopRuntime, LoopState, StagePipeline
from arc_harness.memory.memory import MemoryManager
from arc_harness.runtime.tools import ToolDispatcher
from arc_harness.eval.tracing import Trace, TraceStore
from arc_harness.utils.validation import validate_frame


@dataclass
class EpisodeResult:
    episode_id: str
    status: str
    steps: int
    done: bool
    total_reward: float
    summary: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "status": self.status,
            "steps": self.steps,
            "done": self.done,
            "total_reward": self.total_reward,
            "summary": self.summary,
            "events": self.events,
        }


class EpisodeRunner:
    """Run an episode through a composable stage pipeline."""

    def __init__(
        self,
        memory: MemoryManager,
        hooks: HookManager | None = None,
        checkpoints: CheckpointStore | None = None,
        traces: TraceStore | None = None,
        guardrails: list | None = None,
        pipeline: StagePipeline | None = None,
        delegation: DelegationManager | None = None,
        tools: ToolDispatcher | None = None,
    ) -> None:
        self.memory = memory
        self.hooks = hooks or HookManager()
        self.checkpoints = checkpoints
        self.traces = traces
        self.guardrails = list(guardrails or [])
        self.pipeline = pipeline or StagePipeline()
        self.delegation = delegation
        self.tools = tools

    def run(
        self,
        env: ArcEnvironment,
        agent: ArcAgent,
        max_steps: int = 256,
        episode_id: str | None = None,
        config: RunnerConfig | None = None,
    ) -> EpisodeResult:
        run_config = config or RunnerConfig.from_value(max_steps)
        events: list[dict] = []
        result: EpisodeResult | None = None
        for event in self.run_events(env, agent, episode_id=episode_id, config=run_config):
            events.append(event.to_dict())
            if event.type == "episode.completed":
                result = event.payload["result"]
        if result is None:
            raise RuntimeError("Episode did not produce a completion event.")
        result.events = events
        return result

    def run_events(
        self,
        env: ArcEnvironment,
        agent: ArcAgent,
        max_steps: int = 256,
        episode_id: str | None = None,
        config: RunnerConfig | None = None,
    ) -> Iterator[AgentEvent]:
        run_config = config or RunnerConfig.from_value(max_steps)
        episode_id = episode_id or str(uuid.uuid4())
        validate_environment(env)
        self.memory.reset_episode()
        self.hooks.before_episode(episode_id)
        yield from self._emit(AgentEvent("episode.started", episode_id, {"config": run_config.to_dict()}))
        total_reward = 0.0
        latest: object | None = None
        proposed_action = None
        current_step: int | None = None
        phase = "reset"
        runtime: LoopRuntime | None = None
        trace = Trace(run_config.trace_workflow_name, group_id=episode_id, metadata={"episode_id": episode_id, **run_config.metadata})
        root_span = trace.start_span("episode", metadata={"max_steps": run_config.max_steps})

        try:
            reset_span = trace.start_span("env.reset", parent_id=root_span.span_id)
            latest = env.reset()
            reset_span.finish({"status": getattr(latest, "status", "UNKNOWN")})
            if run_config.validate_frames:
                validate_frame(latest)
            self._run_frame_guardrails(latest, "reset")
            self.memory.working.remember_frame(latest)
            self._checkpoint(run_config, episode_id, -1, latest)
            self.hooks.after_observe(latest)
            yield from self._emit(AgentEvent("frame.observed", episode_id, {"frame": latest.to_dict(include_grid=False)}))
            agent.on_episode_start(self.memory)
            runtime = LoopRuntime(
                env=env,
                agent=agent,
                memory=self.memory,
                hooks=self.hooks,
                guardrails=self.guardrails,
                config=run_config,
                trace=trace,
                root_span=root_span,
                checkpoints=self.checkpoints,
                delegation=self.delegation,
                tools=self.tools,
            )

            for step in range(run_config.max_steps):
                current_step = step
                state = LoopState(
                    episode_id=episode_id,
                    step=step,
                    frame=latest,
                    total_reward=total_reward,
                    metadata={"frames": self.memory.working.frames},
                )
                state = self.pipeline.run_step(state, runtime)
                for event in runtime.drain_events():
                    yield event
                latest = state.frame
                total_reward = state.total_reward
                proposed_action = state.proposed_action
                phase = state.phase
                if state.stop:
                    episode_result = self._finish(
                        agent,
                        episode_id,
                        state.status,
                        len(self.memory.working.steps),
                        state.done,
                        total_reward,
                        run_config,
                        trace=trace,
                        root_span=root_span,
                    )
                    yield from self._emit(AgentEvent("episode.completed", episode_id, {"result": episode_result}))
                    return

            episode_result = self._finish(agent, episode_id, latest.status, len(self.memory.working.steps), False, total_reward, run_config, trace=trace, root_span=root_span)
            yield from self._emit(AgentEvent("episode.completed", episode_id, {"result": episode_result}))
            return
        except BaseException as exc:
            if runtime is not None:
                for event in runtime.drain_events():
                    yield event
            self.hooks.on_error(episode_id, exc)
            latest_frame = latest if hasattr(latest, "to_dict") else None
            error_context = ErrorContext.from_exception(
                episode_id,
                exc,
                step=current_step,
                phase=phase,
                latest_frame=latest_frame,
                proposed_action=proposed_action,
                recent_actions=self.memory.working.recent_actions(),
            )
            self.memory.record_failure(f"{error_context.error_type} during {phase}: {error_context.error}", durable=True)
            error_result = self._finish(
                agent,
                episode_id,
                "ERROR",
                len(self.memory.working.steps),
                False,
                total_reward,
                run_config,
                error_context=error_context.to_dict(),
                trace=trace,
                root_span=root_span,
            )
            yield from self._emit(AgentEvent("episode.error", episode_id, {"error": error_context.to_dict()}))
            yield from self._emit(AgentEvent("episode.completed", episode_id, {"result": error_result}))
            if run_config.abort_on_error:
                raise
            return

    def _finish(
        self,
        agent: ArcAgent,
        episode_id: str,
        status: str,
        steps: int,
        done: bool,
        total_reward: float,
        config: RunnerConfig | None = None,
        error_context: dict | None = None,
        trace: Trace | None = None,
        root_span=None,
    ) -> EpisodeResult:
        agent.on_episode_end(self.memory)
        trace_id = None
        if trace is not None:
            trace_id = trace.trace_id
            if root_span is not None:
                root_span.finish({"status": status, "steps": steps, "done": done})
            trace.finish({"status": status, "steps": steps, "done": done})
            if self.traces is not None and (config is None or config.tracing):
                self.traces.write(trace)
                self.traces.append_jsonl(trace)
        summary = {
            "status": status,
            "steps": steps,
            "done": done,
            "total_reward": total_reward,
            "hypotheses": list(self.memory.working.hypotheses),
            "failures": list(self.memory.working.failures),
            "notes": list(self.memory.working.notes),
            "action_effects": self.memory.working.action_effects(),
            "config": config.to_dict() if config else {},
            "trace_id": trace_id,
        }
        if error_context:
            summary["error"] = error_context
        summary["memories_created"] = self.memory.consolidate_episode(episode_id, summary)
        self.memory.durable.write_episode(episode_id, self.memory.working.steps, summary)
        self.hooks.after_episode(episode_id, summary)
        return EpisodeResult(episode_id=episode_id, status=status, steps=steps, done=done, total_reward=total_reward, summary=summary)

    def _emit(self, event: AgentEvent) -> Iterator[AgentEvent]:
        self.hooks.emit(event)
        yield event

    def _checkpoint(self, config: RunnerConfig, episode_id: str, step: int, frame) -> None:
        if config.checkpoint and self.checkpoints is not None:
            self.checkpoints.write(episode_id, step, frame, self.memory.working.steps, extra={"config": config.to_dict()})

    def _run_frame_guardrails(self, frame, phase: str) -> None:
        for guardrail in self.guardrails:
            method = getattr(guardrail, "check_frame", None)
            if not method:
                continue
            result = method(frame, phase)
            if result.decision == GuardrailDecision.FAIL:
                raise RuntimeError(f"Frame guardrail failed: {result.reason}")
