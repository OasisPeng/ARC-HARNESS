from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arc_harness import (
    Action,
    ActionBudgetHook,
    ActionType,
    ArcThread,
    CallableModel,
    ContextBudget,
    ContextManager,
    CoordinateBoundsGuardrail,
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
    JsonPolicyModel,
    MemoryPolicy,
    ModelBackedAgent,
    ModelInput,
    ModelOutput,
    OfficialArcEnvironment,
    OfficialSmokeRunner,
    RunnerConfig,
    RuleLearningAgent,
    SubAgentResult,
    build_submission_manifest,
    check_kaggle_readiness,
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


class OutOfBoundsAgent(HeuristicAgent):
    def choose_action(self, frames, latest_frame, memory):
        return Action(ActionType.ACTION6, (99, 99))


class TargetAgent(HeuristicAgent):
    def choose_action(self, frames, latest_frame, memory):
        return Action(ActionType.ACTION6, (1, 0))


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
            manager = ContextManager(ContextBudget(max_tokens=320, memory_tokens=90, recent_step_tokens=80, trace_tokens=90))
            bundle = thread.build_context(
                latest_frame=latest,
                trace_id=result.summary["trace_id"],
                query="changed cells reward target",
                manager=manager,
            )
            rendered = bundle.render()
            self.assertLessEqual(bundle.total_tokens, 320)
            self.assertIn("memory-policy", rendered)
            self.assertIn("recent_steps", rendered)
            self.assertIn("Latest Frame Summary", rendered)
            self.assertIn("Trace Summary", rendered)

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

    def test_invalid_action_records_error_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thread = ArcThread(memory_dir=tmp)
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


if __name__ == "__main__":
    unittest.main()
