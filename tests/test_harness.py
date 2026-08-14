from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from arc_harness import (
    Action,
    ActionBudgetHook,
    ActionType,
    ArcThread,
    CallableModel,
    CapabilityError,
    CapabilityRegistry,
    ContextBudget,
    ContextManager,
    CoordinateBoundsGuardrail,
    DEFAULT_CAPABILITY_REGISTRY,
    DefaultRecoveryPolicy,
    KaggleReadinessReport,
    DelegatingPlannerAgent,
    DelegationConfig,
    DelegationManager,
    EnvironmentResult,
    EvalCase,
    EvaluationRunner,
    Frame,
    HandoffAgent,
    HandoffRule,
    HeuristicAgent,
    HookDecision,
    HookMatcher,
    KagglePackage,
    JsonPolicyModel,
    MemoryPolicy,
    ModelBackedAgent,
    ModelInput,
    ModelOutput,
    NoRecoveryPolicy,
    OfficialArcEnvironment,
    OfficialSmokeRunner,
    LocalSubprocessSandbox,
    LoopRuntime,
    LoopState,
    ProviderDescriptor,
    RunnerConfig,
    RuleLearningAgent,
    SandboxCommand,
    SandboxPolicy,
    SandboxPolicyError,
    StagePipeline,
    SubAgentResult,
    ToolCall,
    ToolContext,
    ToolDispatcher,
    ToolError,
    ToolRegistry,
    ToolSpec,
    ToolUseStage,
    build_kaggle_package,
    build_submission_manifest,
    check_kaggle_readiness,
    delegating_planner_loop_stages,
)
from arc_harness.adapters import KaggleAgentAdapter
from arc_harness.models import load_model_from_config
from arc_harness.official import EnvironmentFileCatalog, coerce_official_frame
from arc_harness.submission import build_adapter


class TinyEnv:
    def __init__(self) -> None:
        self.grid = [[0, 1], [0, 2]]

    def reset(self) -> Frame:
        return Frame.from_grid(self.grid)

    def step(self, action: Action) -> EnvironmentResult:
        if action.kind == ActionType.ACTION6 and action.xy == (1, 0):
            self.grid[0][1] = 3
            return EnvironmentResult(Frame.from_grid(self.grid, status="WIN"), reward=1.0, done=True)
        return EnvironmentResult(Frame.from_grid(self.grid), reward=0.0, done=False)


class RewriteHook:
    def before_action(self, step: int, frame: Frame, action: Action) -> HookDecision:
        return HookDecision.rewrite(Action(ActionType.ACTION6, (1, 0)), "force target")


class BadActionAgent(HeuristicAgent):
    def choose_action(self, frames, latest_frame, memory):
        return Action("BAD_ACTION")


class NoopAgent(HeuristicAgent):
    def choose_action(self, frames, latest_frame, memory):
        return Action(ActionType.ACTION1)


class OutOfBoundsAgent(HeuristicAgent):
    def choose_action(self, frames, latest_frame, memory):
        return Action(ActionType.ACTION6, (99, 99))


class TargetAgent(HeuristicAgent):
    def choose_action(self, frames, latest_frame, memory):
        return Action(ActionType.ACTION6, (1, 0))


class ToolPlanningAgent(TargetAgent):
    def choose_tools(self, frames, latest_frame, memory, tools):
        return [
            ToolCall("observe_objects", {"limit": 4}),
            {"name": "propose_actions", "arguments": {"limit": 2}},
            {"name": "write_note", "arguments": {"text": "tool-assisted step"}},
        ]


class CountingHook:
    def __init__(self) -> None:
        self.count = 0

    def before_action(self, step: int, frame: Frame, action: Action):
        self.count += 1
        return action


class FlakyPerceptionSubAgent:
    name = "FlakyPerceptionSubAgent"
    kinds = ("flaky_perceive",)

    def __init__(self) -> None:
        self.calls = 0

    def run(self, task, memory):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return SubAgentResult(task.task_id, self.name, True, {"calls": self.calls}, "recovered", confidence=0.7)


class FakeCapabilityProvider:
    descriptor = ProviderDescriptor(
        capability="model",
        name="fake",
        version="1.0",
        supports=("predict", "offline"),
        metadata={"purpose": "test"},
    )


class NoteStage:
    name = "note"

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        runtime.memory.add_note(f"custom stage at step {state.step}")
        state.metadata["custom_stage_seen"] = True
        runtime.emit("custom.stage", state.episode_id, {"step": state.step})
        return state


class FlakyPerceptionStage:
    name = "perception"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary perception miss")
        runtime.memory.add_note("flaky perception recovered")
        return state


class FakeOfficialAction:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeOfficialFrame:
    def __init__(self, grid, state="NOT_FINISHED") -> None:
        self.grid = grid
        self.state = state


class FakeOfficialEnv:
    def __init__(self) -> None:
        self.grid = [[0, 1], [0, 2]]
        self.action_space = [FakeOfficialAction("ACTION1"), FakeOfficialAction("ACTION6")]
        self.calls = []

    def reset(self):
        return FakeOfficialFrame(self.grid)

    def step(self, action, data=None, reasoning=None):
        self.calls.append({"action": action.name, "data": data, "reasoning": reasoning})
        if action.name == "ACTION6" and data == {"x": 1, "y": 0}:
            self.grid[0][1] = 3
            return FakeOfficialFrame(self.grid, state="WIN")
        return FakeOfficialFrame(self.grid)


class HarnessTests(unittest.TestCase):
    def test_capability_registry_registers_and_requires_providers(self) -> None:
        registry = CapabilityRegistry()
        provider = FakeCapabilityProvider()
        returned = registry.register(provider)
        self.assertIs(returned, provider)
        self.assertIs(registry.get("model", "fake"), provider)
        self.assertIs(registry.require("model", "fake", supports=("predict",)), provider)
        self.assertEqual(registry.capabilities(), ("model",))
        self.assertEqual(registry.list("model")[0].metadata["purpose"], "test")

        with self.assertRaises(CapabilityError):
            registry.register(provider)
        with self.assertRaises(CapabilityError):
            registry.require("sandbox", "missing")
        with self.assertRaises(CapabilityError):
            registry.require("model", "fake", supports=("stream",))

    def test_default_capability_registry_exposes_local_sandbox(self) -> None:
        sandbox = DEFAULT_CAPABILITY_REGISTRY.require("sandbox", "local_subprocess", supports=("timeout",))
        self.assertIsInstance(sandbox, LocalSubprocessSandbox)

    def test_tool_registry_dispatches_and_enforces_policy(self) -> None:
        def echo(arguments, context):
            return {"echo": arguments["text"]}

        registry = ToolRegistry()
        registry.register(echo, ToolSpec("echo", "Echo text.", {"type": "object", "properties": {"text": {"type": "string"}}}, required=("text",)))
        dispatcher = ToolDispatcher(registry)
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            context = ToolContext(memory=thread.memory, frame=Frame.from_grid([[1]]))
            result = dispatcher.dispatch(ToolCall("echo", {"text": "ok"}), context)
            self.assertTrue(result.ok)
            self.assertEqual(result.output["echo"], "ok")
            missing = dispatcher.dispatch(ToolCall("echo", {}), context)
            self.assertFalse(missing.ok)
            denied = ToolDispatcher(registry, permissions={"echo": "deny"}).dispatch(ToolCall("echo", {"text": "no"}), context)
            self.assertFalse(denied.ok)
            self.assertIn("denied", denied.error)

    def test_local_sandbox_runs_command_and_captures_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = LocalSubprocessSandbox(
                SandboxPolicy(
                    allowed_commands=(Path(sys.executable).name,),
                    allowed_cwds=(tmp,),
                    max_output_chars=200,
                )
            )
            result = sandbox.run([sys.executable, "-c", "print('arc sandbox ok')"], cwd=tmp)
            self.assertTrue(result.ok)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("arc sandbox ok", result.stdout)
            self.assertEqual(result.metadata["cwd"], str(Path(tmp).resolve()))

    def test_thread_exposes_builtin_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            self.assertIn("observe_objects", thread.available_tools())
            frame = Frame.from_grid([[1, 1, 0], [0, 2, 2]])
            objects = thread.run_tool({"name": "observe_objects", "arguments": {"limit": 4}}, frame=frame)
            self.assertTrue(objects.ok)
            self.assertEqual(objects.output["nonzero_object_count"], 2)
            actions = thread.run_tool({"name": "propose_actions", "arguments": {"limit": 2}}, frame=frame)
            self.assertTrue(actions.ok)
            self.assertEqual(len(actions.output["actions"]), 2)
            note = thread.run_tool({"name": "write_note", "arguments": {"text": "remember this"}}, frame=frame)
            self.assertTrue(note.ok)
            self.assertIn("remember this", thread.memory.working.notes)

    def test_local_sandbox_rejects_policy_violations(self) -> None:
        sandbox = LocalSubprocessSandbox(SandboxPolicy(allowed_commands=("python",)))
        with self.assertRaises(SandboxPolicyError):
            sandbox.run("python -c 'print(1)'")
        with self.assertRaises(SandboxPolicyError):
            sandbox.run(["rm", "-rf", "/tmp/not-real"])
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            policy = SandboxPolicy(allowed_commands=(Path(sys.executable).name,), allowed_cwds=(Path(tmp) / "inside",))
            scoped = LocalSubprocessSandbox(policy)
            with self.assertRaises(SandboxPolicyError):
                scoped.run([sys.executable, "-c", "print(1)"], cwd=outside)

    def test_local_sandbox_returns_timeout_result(self) -> None:
        sandbox = LocalSubprocessSandbox(SandboxPolicy(allowed_commands=(Path(sys.executable).name,), timeout_seconds=0.1))
        result = sandbox.run(SandboxCommand([sys.executable, "-c", "import time; time.sleep(1)"]))
        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)

    def test_thread_runs_and_persists_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            result = thread.run_episode(TinyEnv(), HeuristicAgent(), max_steps=8)
            self.assertTrue(result.done)
            self.assertEqual(result.status, "WIN")

            resumed = ArcThread.resume(thread.thread_id, memory_dir=tmp)
            self.assertEqual(len(resumed.history), 1)
            listed = ArcThread.list_threads(memory_dir=tmp)
            self.assertEqual(listed[0]["thread_id"], thread.thread_id)
            state = ArcThread.read_thread(thread.thread_id, memory_dir=tmp)
            self.assertEqual(state["thread_id"], thread.thread_id)

    def test_adapter_returns_competition_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = KaggleAgentAdapter(HeuristicAgent(), memory_dir=Path(tmp) / "memory")
            action = adapter.choose_action([[[0, 1], [0, 2]]], [[0, 1], [0, 2]])
            self.assertIsInstance(action, tuple)
            self.assertEqual(action[0], "ACTION6")

    def test_streamed_events_and_episode_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp, metadata={"purpose": "test"})
            config = RunnerConfig(max_steps=8, loop_window=2, metadata={"game": "tiny"})
            events = list(thread.run_streamed(TinyEnv(), RuleLearningAgent(), config=config))
            self.assertEqual(events[0]["type"], "episode.started")
            self.assertEqual(events[0]["payload"]["config"]["metadata"]["game"], "tiny")
            self.assertEqual(events[-1]["type"], "episode.completed")
            self.assertIn("stage.started", [event["type"] for event in events])
            self.assertIn("stage.completed", [event["type"] for event in events])
            episode_id = events[-1]["episode_id"]
            replay = thread.read_episode(episode_id)
            self.assertEqual(replay[0]["type"], "summary")
            typed_replay = thread.load_replay(episode_id)
            self.assertEqual(typed_replay.status, "WIN")
            self.assertTrue(typed_replay.done)
            self.assertGreaterEqual(len(typed_replay.steps), 1)
            checkpoint = thread.read_checkpoint(episode_id)
            self.assertEqual(checkpoint["episode_id"], episode_id)
            self.assertGreaterEqual(checkpoint["step"], 0)
            trace_id = events[-1]["payload"]["result"]["summary"]["trace_id"]
            trace = thread.read_trace(trace_id)
            self.assertEqual(trace["group_id"], episode_id)
            self.assertGreaterEqual(len(trace["spans"]), 3)
            replay_markdown = thread.replay_markdown(episode_id)
            self.assertIn("| step | action | reward | changed_cells | status | progressed |", replay_markdown)
            self.assertIn("ACTION6", replay_markdown)
            trace_timeline = thread.trace_timeline(trace_id)
            self.assertEqual(trace_timeline.group_id, episode_id)
            self.assertIn("episode", trace_timeline.stage_counts())
            self.assertIn("| span | duration_ms | parent | metadata |", trace_timeline.to_markdown())
            replay_report = Path(tmp) / "reports" / "replay.md"
            trace_report = Path(tmp) / "reports" / "trace.md"
            thread.write_replay_report(episode_id, replay_report)
            thread.write_trace_report(trace_id, trace_report)
            self.assertTrue(replay_report.exists())
            self.assertTrue(trace_report.exists())

    def test_stage_pipeline_exposes_order_and_allows_insertion(self) -> None:
        pipeline = StagePipeline()
        self.assertEqual(pipeline.names(), ("done_check", "context.build", "tool.use", "decision", "permission", "action.execute", "stop_check"))
        pipeline.insert_before("decision", NoteStage())
        self.assertEqual(pipeline.names()[3], "note")

        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp, pipeline=pipeline)
            events = list(thread.run_streamed(TinyEnv(), HeuristicAgent(), config=RunnerConfig(max_steps=4)))
            self.assertIn("custom.stage", [event["type"] for event in events])
            context_events = [event for event in events if event["type"] == "context.built"]
            self.assertTrue(context_events)
            self.assertIn("action_map", context_events[0]["payload"]["context"]["sections"])
            self.assertIn("custom stage at step 0", thread.memory.working.notes)
            self.assertEqual(events[-1]["payload"]["result"]["status"], "WIN")

    def test_tool_use_stage_dispatches_agent_tools_before_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            events = list(thread.run_streamed(TinyEnv(), ToolPlanningAgent(), config=RunnerConfig(max_steps=2)))
            event_types = [event["type"] for event in events]
            self.assertIn("tool.requested", event_types)
            self.assertIn("tool.completed", event_types)
            self.assertEqual(events[-1]["payload"]["result"]["status"], "WIN")
            self.assertIn("tool-assisted step", thread.memory.working.notes)
            completed = [event for event in events if event["type"] == "tool.completed"]
            self.assertEqual(len(completed), 3)
            trace_id = events[-1]["payload"]["result"]["summary"]["trace_id"]
            trace = thread.read_trace(trace_id)
            self.assertIn("tool.dispatch", [span["name"] for span in trace["spans"]])

    def test_tool_use_stage_records_failed_tool_result(self) -> None:
        class MissingToolAgent(TargetAgent):
            def choose_tools(self, frames, latest_frame, memory, tools):
                return [ToolCall("missing_tool", {})]

        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            events = list(thread.run_streamed(TinyEnv(), MissingToolAgent(), config=RunnerConfig(max_steps=2)))
            self.assertIn("tool.failed", [event["type"] for event in events])
            self.assertTrue(any("missing_tool" in failure for failure in thread.memory.working.failures))
            self.assertEqual(events[-1]["payload"]["result"]["status"], "WIN")

    def test_recovery_policy_retries_failed_stage(self) -> None:
        stage = FlakyPerceptionStage()
        pipeline = StagePipeline(recovery_policy=DefaultRecoveryPolicy(max_retries=1))
        pipeline.insert_before("decision", stage)
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp, pipeline=pipeline)
            events = list(thread.run_streamed(TinyEnv(), HeuristicAgent(), config=RunnerConfig(max_steps=4)))
            self.assertGreaterEqual(stage.calls, 2)
            self.assertIn("recovery.triggered", [event["type"] for event in events])
            self.assertIn("flaky perception recovered", thread.memory.working.notes)
            self.assertEqual(events[-1]["payload"]["result"]["status"], "WIN")

    def test_delegating_planner_pipeline_runs_perception_exploration_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            delegation = DelegationManager.with_default_subagents(config=DelegationConfig(parallel_workers=2))
            pipeline = StagePipeline(delegating_planner_loop_stages(delegation_config=DelegationConfig(max_retries=1)))
            thread = ArcThread(memory_dir=tmp, delegation=delegation, pipeline=pipeline)
            events = list(thread.run_streamed(TinyEnv(), HeuristicAgent(), config=RunnerConfig(max_steps=4)))
            event_types = [event["type"] for event in events]
            self.assertIn("perception.completed", event_types)
            self.assertIn("exploration.completed", event_types)
            self.assertIn("plan.created", event_types)
            self.assertEqual(events[-1]["payload"]["result"]["status"], "WIN")
            self.assertGreaterEqual(len([event for event in thread.read_delegation_events() if event["type"] == "subtask.completed"]), 3)

    def test_action_budget_hook_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp, hooks=[ActionBudgetHook(0)])
            result = thread.run_episode(TinyEnv(), HeuristicAgent(), max_steps=8)
            self.assertEqual(result.status, "BLOCKED")
            self.assertFalse(result.done)

    def test_rewrite_hook_emits_rewritten_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp, hooks=[RewriteHook()])
            events = list(thread.run_streamed(TinyEnv(), HeuristicAgent(), max_steps=8))
            self.assertIn("action.rewritten", [event["type"] for event in events])
            self.assertEqual(events[-1]["payload"]["result"]["status"], "WIN")

    def test_memory_skill_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            path = thread.memory.durable.save_skill("Analyze New Game", "Inspect diffs before planning.", description="ARC game analysis")
            self.assertTrue(path.exists())
            self.assertIn("Inspect diffs", thread.memory.durable.read_skill("Analyze New Game"))

    def test_structured_memory_search_filters_category_and_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            thread.memory.add_fact(
                "ACTION6 click near target changed the blue object.",
                category="insight",
                namespace=("action-effects",),
                tags=("action-effect", "action6"),
                importance=0.8,
            )
            thread.memory.add_fact(
                "Use row scanning before random exploration.",
                category="procedure",
                namespace=("procedures",),
                tags=("planning",),
            )
            hits = thread.memory.search_entries("blue target click", namespace=("action-effects",), category="insight", limit=3)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].category, "insight")
            self.assertIn("ACTION6", hits[0].text)

    def test_episode_consolidation_creates_searchable_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            result = thread.run_episode(TinyEnv(), RuleLearningAgent(), config=RunnerConfig(max_steps=8))
            self.assertGreater(result.summary["memories_created"], 0)
            hits = thread.memory.search_entries("changed cells reward", category="insight", limit=5)
            self.assertGreaterEqual(len(hits), 1)
            self.assertEqual(hits[0].metadata["status"], "WIN")

    def test_memory_policy_renders_compact_guidance(self) -> None:
        policy = MemoryPolicy()
        self.assertIn("<memory-policy>", policy.render())
        self.assertIn("current frame", policy.render())

    def test_context_manager_builds_budgeted_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            result = thread.run_episode(TinyEnv(), RuleLearningAgent(), config=RunnerConfig(max_steps=8))
            latest = thread.memory.working.frames[-1]
            manager = ContextManager(ContextBudget(max_tokens=720, memory_tokens=120, recent_step_tokens=90, trace_tokens=120))
            bundle = thread.build_context(
                latest_frame=latest,
                trace_id=result.summary["trace_id"],
                query="changed cells reward target",
                manager=manager,
            )
            rendered = bundle.render()
            self.assertLessEqual(bundle.total_tokens, manager.budget.max_tokens)
            self.assertIn("memory-policy", rendered)
            self.assertIn("recent_steps", rendered)
            self.assertIn("Latest Frame Summary", rendered)
            self.assertIn("Trace Summary", rendered)
            self.assertIn("Object Summary", rendered)
            self.assertIn("Tried And Failed Action Map", rendered)

    def test_context_injector_returns_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            thread.memory.add_fact("Blue target changes after ACTION6 click.", category="insight", namespace=("action-effects",))
            text = thread.inject_context(query="blue action6 target")
            self.assertIn("Relevant Durable Memory", text)
            self.assertIn("Blue target", text)

    def test_invalid_runner_config_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            RunnerConfig(max_steps=0)

    def test_invalid_action_recovers_with_fallback_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            events = list(thread.run_streamed(TinyEnv(), BadActionAgent(), config=RunnerConfig(max_steps=4, abort_on_error=False)))
            result = events[-1]["payload"]["result"]
            self.assertEqual(result["status"], "WIN")
            self.assertIn("recovery.triggered", [event["type"] for event in events])
            self.assertIn("fallback_action", [event["payload"]["decision"]["kind"] for event in events if event["type"] == "recovery.triggered"])

    def test_no_recovery_policy_preserves_error_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp, pipeline=StagePipeline(recovery_policy=NoRecoveryPolicy()))
            result = thread.run_episode(TinyEnv(), BadActionAgent(), config=RunnerConfig(max_steps=4, abort_on_error=False))
            self.assertEqual(result.status, "ERROR")
            self.assertIn("error", result.summary)
            self.assertEqual(result.summary["error"]["error_type"], "ValidationError")
            replay = thread.load_replay(result.episode_id)
            self.assertEqual(replay.status, "ERROR")

    def test_hook_matcher_filters_action_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hook = CountingHook()
            thread = ArcThread(memory_dir=tmp, hooks=[HookMatcher(hook=hook, event="before_action", action="ACTION1")])
            thread.run_episode(TinyEnv(), HeuristicAgent(), config=RunnerConfig(max_steps=4))
            self.assertEqual(hook.count, 0)

            hook2 = CountingHook()
            thread2 = ArcThread(memory_dir=tmp, hooks=[HookMatcher(hook=hook2, event="before_action", action="ACTION6")])
            thread2.run_episode(TinyEnv(), HeuristicAgent(), config=RunnerConfig(max_steps=4))
            self.assertGreater(hook2.count, 0)

    def test_guardrail_failure_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp, guardrails=[CoordinateBoundsGuardrail()])
            result = thread.run_episode(
                TinyEnv(),
                OutOfBoundsAgent(),
                config=RunnerConfig(max_steps=4, validate_actions=False, abort_on_error=False),
            )
            self.assertEqual(result.status, "ERROR")
            self.assertIn("Action guardrail failed", result.summary["error"]["error"])

    def test_evaluation_runner_summarizes_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = EvaluationRunner(memory_dir=tmp)
            report = runner.run(
                [
                    EvalCase("tiny-1", TinyEnv),
                    EvalCase("tiny-2", TinyEnv),
                ],
                HeuristicAgent,
                config=RunnerConfig(max_steps=8),
            )
            self.assertEqual(report.total, 2)
            self.assertEqual(report.completed, 2)
            self.assertEqual(report.status_counts(), {"WIN": 2})
            self.assertEqual(report.failure_counts(), {"completed": 2})

    def test_evaluation_runner_records_failure_taxonomy_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = EvaluationRunner(memory_dir=tmp)
            report = runner.run(
                [EvalCase("stuck", TinyEnv)],
                NoopAgent,
                config=RunnerConfig(max_steps=2),
            )
            result = report.results[0]
            self.assertEqual(result.failure_reason, "max_steps_exceeded")
            self.assertEqual(result.metrics["noop_action_count"], 2)
            self.assertTrue(Path(result.replay_path).exists())
            self.assertTrue(Path(result.trace_path).exists())
            self.assertIn("failure_counts", report.to_dict())
            self.assertIn("max_steps_exceeded", report.to_markdown())
            report_json = report.write_json(Path(tmp) / "eval.json")
            report_md = report.write_markdown(Path(tmp) / "eval.md")
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())

            winning = EvaluationRunner(memory_dir=Path(tmp) / "winning").run(
                [EvalCase("win", TinyEnv)],
                TargetAgent,
                config=RunnerConfig(max_steps=2),
            )
            comparison = winning.compare(report)
            self.assertGreater(comparison["completion_rate_delta"], 0)

    def test_default_subagents_perceive_frame_and_record_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            result = thread.delegate("perceive", {"frame": Frame.from_grid([[1, 1, 0], [0, 2, 2]])})
            self.assertTrue(result.ok)
            self.assertEqual(result.agent_name, "PerceptionSubAgent")
            self.assertEqual(result.output["colors"], [0, 1, 2])
            self.assertGreaterEqual(result.output["component_count"], 3)
            hits = thread.memory.search_entries("PerceptionSubAgent components", namespace=("delegation", "perceive"), limit=5)
            self.assertGreaterEqual(len(hits), 1)

    def test_diff_subagent_summarizes_action_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            before = Frame.from_grid([[0, 1], [0, 2]])
            after = Frame.from_grid([[0, 3], [0, 2]], status="WIN")
            result = thread.delegate("diff", {"before": before, "after": after, "action": Action(ActionType.ACTION6, (1, 0))})
            self.assertTrue(result.ok)
            self.assertEqual(result.output["changed_count"], 1)
            self.assertTrue(result.output["status_changed"])
            self.assertEqual(result.output["bbox"], {"x1": 1, "y1": 0, "x2": 1, "y2": 0})

    def test_explorer_subagent_avoids_tried_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            result = thread.delegate(
                "explore",
                {
                    "frame": Frame.from_grid([[0, 1], [0, 2]]),
                    "tried_actions": [Action(ActionType.ACTION6, (0, 0)), Action(ActionType.ACTION1)],
                },
                budget=3,
            )
            values = result.output["competition_values"]
            self.assertEqual(len(values), 3)
            self.assertNotIn(("ACTION6", 0, 0), values)
            self.assertNotIn("ACTION1", values)

    def test_delegation_retry_records_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = DelegationManager([FlakyPerceptionSubAgent()], config=DelegationConfig(max_retries=1))
            thread = ArcThread(memory_dir=tmp, delegation=manager)
            result = thread.delegate("flaky_perceive", {"frame": Frame.from_grid([[0]])})
            self.assertTrue(result.ok)
            self.assertEqual(result.output["calls"], 2)
            event_types = [event["type"] for event in thread.read_delegation_events()]
            self.assertEqual(event_types, ["subtask.started", "subtask.retrying", "subtask.completed"])

    def test_delegate_many_runs_multiple_subtasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            results = thread.delegate_many(
                [
                    ("perceive", {"frame": Frame.from_grid([[1, 0], [0, 2]])}),
                    ("explore", {"frame": Frame.from_grid([[1, 0], [0, 2]]), "tried_actions": []}),
                ],
                config=DelegationConfig(parallel_workers=2),
            )
            self.assertEqual([result.agent_name for result in results], ["PerceptionSubAgent", "ExplorerSubAgent"])
            self.assertEqual(len([event for event in thread.read_delegation_events() if event["type"] == "subtask.completed"]), 2)

    def test_planner_subagent_ranks_action_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            perception = thread.delegate("perceive", {"frame": Frame.from_grid([[0, 1], [0, 0]])})
            result = thread.delegate(
                "plan",
                {
                    "frame": Frame.from_grid([[0, 1], [0, 0]]),
                    "perception": perception.output,
                    "candidate_actions": [Action(ActionType.ACTION6, (0, 0)), Action(ActionType.ACTION6, (1, 0))],
                },
                budget=2,
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.agent_name, "PlannerSubAgent")
            self.assertEqual(result.output["stop_reason"], "plan_ready")
            self.assertEqual(result.output["plan"][0]["action"]["xy"], (1, 0))

    def test_delegation_trace_records_subagent_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
            thread.delegate("perceive", {"frame": Frame.from_grid([[1, 0], [0, 2]])})
            trace = thread.read_delegation_trace()
            self.assertEqual(trace["workflow_name"], "ARC delegation")
            self.assertEqual(trace["group_id"], thread.thread_id)
            self.assertEqual(trace["spans"][0]["name"], "subagent.perceive")
            self.assertEqual(trace["spans"][0]["metadata"]["event"], "subtask.completed")

    def test_delegating_planner_agent_closes_loop_with_subagents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            delegation = DelegationManager.with_default_subagents(config=DelegationConfig(parallel_workers=2))
            agent = DelegatingPlannerAgent(delegation=delegation)
            thread = ArcThread(memory_dir=tmp, delegation=delegation)
            result = thread.run_episode(TinyEnv(), agent, config=RunnerConfig(max_steps=4))
            self.assertTrue(result.done)
            self.assertEqual(result.status, "WIN")
            self.assertGreaterEqual(len(delegation.events), 3)

    def test_handoff_agent_transfers_control_to_specialist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rule = HandoffRule(
                target="target",
                reason="use specialist after the first miss",
                predicate=lambda state, memory: state["step"] >= 1,
            )
            agent = HandoffAgent(
                primary=HeuristicAgent(actions=[ActionType.ACTION1]),
                specialists={"target": TargetAgent()},
                rules=[rule],
            )
            thread = ArcThread(memory_dir=tmp)
            result = thread.run_episode(TinyEnv(), agent, config=RunnerConfig(max_steps=4))
            self.assertTrue(result.done)
            self.assertEqual(result.status, "WIN")
            self.assertEqual(agent.active_agent_name, "target")
            hits = thread.memory.search("Handoff route target", limit=3)
            self.assertGreaterEqual(len(hits), 1)

    def test_official_frame_coercion_accepts_object_and_dict_shapes(self) -> None:
        frame = coerce_official_frame(FakeOfficialFrame([[1, 2]], state="WIN"))
        self.assertEqual(frame.grid, ((1, 2),))
        self.assertEqual(frame.status, "WIN")

        frame2 = coerce_official_frame({"frame": [[0]], "state": "GAME_OVER"})
        self.assertEqual(frame2.status, "GAME_OVER")

    def test_official_environment_wraps_reset_and_complex_step(self) -> None:
        env = FakeOfficialEnv()
        wrapped = OfficialArcEnvironment(env, game_id="fake-game")
        start = wrapped.reset()
        self.assertEqual(start.status, "NOT_FINISHED")
        result = wrapped.step(Action(ActionType.ACTION6, (1, 0)))
        self.assertTrue(result.done)
        self.assertEqual(result.frame.status, "WIN")
        self.assertEqual(env.calls[-1]["data"], {"x": 1, "y": 0})
        self.assertEqual(result.info["official_action"], "ACTION6")

    def test_environment_file_catalog_lists_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game_dir = Path(tmp) / "environment_files" / "demo"
            game_dir.mkdir(parents=True)
            (game_dir / "metadata.json").write_text('{"game_id": "demo01", "title": "Demo"}', encoding="utf-8")
            catalog = EnvironmentFileCatalog(Path(tmp) / "environment_files")
            games = catalog.list_games()
            self.assertEqual(games[0]["game_id"], "demo01")
            self.assertEqual(games[0]["metadata"]["title"], "Demo")

    def test_json_policy_model_and_model_backed_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            policy_path.write_text('{"0,1|0,2": ["ACTION6", 1, 0]}', encoding="utf-8")
            agent = ModelBackedAgent(JsonPolicyModel(policy_path), inject_context=False)
            thread = ArcThread(memory_dir=tmp)
            result = thread.run_episode(TinyEnv(), agent, config=RunnerConfig(max_steps=2))
            self.assertTrue(result.done)
            self.assertEqual(result.status, "WIN")

    def test_callable_model_can_return_plan_for_kaggle_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = CallableModel(
                lambda model_input: ModelOutput(plan=[{"action": {"kind": "ACTION6", "xy": (1, 0)}, "score": 1.0}]),
                name="PlanModel",
            )
            adapter = KaggleAgentAdapter(ModelBackedAgent(model, inject_context=False), memory_dir=Path(tmp) / "memory")
            action = adapter.choose_action([[[0, 1], [0, 2]]], [[0, 1], [0, 2]])
            self.assertEqual(action, ("ACTION6", 1, 0))

    def test_model_backed_agent_receives_compressed_arc_context(self) -> None:
        captured = {}

        def choose(model_input: ModelInput) -> ModelOutput:
            captured["context"] = model_input.context
            return ModelOutput(action=Action(ActionType.ACTION6, (1, 0)))

        with tempfile.TemporaryDirectory() as tmp:
            agent = ModelBackedAgent(CallableModel(choose, name="ContextAwareModel"))
            thread = ArcThread(memory_dir=tmp)
            result = thread.run_episode(TinyEnv(), agent, config=RunnerConfig(max_steps=2))
            self.assertEqual(result.status, "WIN")
            self.assertIn("Object Summary", captured["context"])
            self.assertIn("Tried And Failed Action Map", captured["context"])

    def test_model_config_loads_json_policy_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            config_path = Path(tmp) / "model.json"
            policy_path.write_text('{"*": "ACTION1"}', encoding="utf-8")
            config_path.write_text(f'{{"type": "json_policy", "path": "{policy_path}"}}', encoding="utf-8")
            model = load_model_from_config(config_path)
            output = model.predict(ModelInput([], Frame.from_grid([[0]])))
            self.assertEqual(output.best_action().kind, "ACTION1")

    def test_submission_helpers_build_adapter_from_explicit_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = build_adapter(HeuristicAgent(), memory_dir=Path(tmp) / "memory")
            action = adapter.choose_action([[[0, 1], [0, 2]]], [[0, 1], [0, 2]])
            self.assertEqual(action, ("ACTION6", 0, 0))

    def test_official_smoke_runner_skips_when_official_package_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_root = Path(tmp) / "environment_files"
            game_dir = env_root / "demo"
            game_dir.mkdir(parents=True)
            (game_dir / "metadata.json").write_text('{"game_id": "demo"}', encoding="utf-8")
            report = OfficialSmokeRunner(env_root, memory_dir=Path(tmp) / "memory").run(HeuristicAgent(), max_games=1)
            self.assertEqual(report.total, 1)
            self.assertIn(report.results[0].status, {"SKIPPED", "ERROR"})
            self.assertTrue(report.results[0].error)

    def test_kaggle_readiness_report_checks_manifest_and_submission(self) -> None:
        report = check_kaggle_readiness(package_root="arc_harness", agent=HeuristicAgent())
        self.assertIsInstance(report, KaggleReadinessReport)
        names = [check.name for check in report.checks]
        self.assertIn("package_files", names)
        self.assertIn("submission_functions", names)
        submission = [check for check in report.checks if check.name == "submission_functions"][0]
        self.assertTrue(submission.ok)
        self.assertIn("arc_harness/submission.py", build_submission_manifest("arc_harness"))

    def test_build_kaggle_package_copies_manifest_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = build_kaggle_package(Path(tmp) / "bundle")
            self.assertIsInstance(package, KagglePackage)
            output = Path(package.output_dir)
            self.assertTrue((output / "arc_harness" / "submission.py").exists())
            self.assertTrue((output / "submission.py").exists())
            self.assertTrue(Path(package.manifest_path).exists())
            self.assertIn("arc_harness/kaggle.py", package.files)


if __name__ == "__main__":
    unittest.main()
