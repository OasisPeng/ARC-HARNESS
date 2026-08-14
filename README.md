# ARC Agent Harness

Version: `0.4.0`

This project is a small Python harness for ARC-AGI-3-style interactive agents.
It borrows three architectural ideas while staying offline-friendly:

- Claude Code SDK style: an explicit observe -> decide -> act -> observe loop with hooks.
- Hermes / pi-hermes style: layered memory for facts, episodes, failures, and procedures.
- Codex SDK style: a simple Thread API that can run, stream, and resume episodes.

It is intentionally dependency-free so the core can be copied into a Kaggle
Notebook or packaged as a small module.

## Layout

```text
arc_harness/
  actions.py       Action and frame data structures.
  agent.py         Base ARC agent interface.
  capabilities.py  Runtime capability registry and provider descriptors.
  environment.py   Minimal environment protocol.
  errors.py        Structured error context and validation exceptions.
  events.py        Structured event stream.
  evaluation.py    Batch evaluation runner and JSON-friendly reports.
  guardrails.py    Action/frame/result guardrail interfaces.
  hooks.py         Hook interface and built-in logging hook.
  kaggle.py        Kaggle readiness checks and submission manifest.
  loop.py          EpisodeRunner: the agent loop.
  loop_stages.py   Stage-based loop state, runtime, and pipeline.
  memory.py        Layered memory stores.
  memory_policy.py Policy-only guidance for when/how to retrieve memory.
  memory_store.py  SQLite memory store with hybrid search.
  models.py        Offline local model protocol and model-backed agent.
  official.py      Optional official ARC-AGI-3 toolkit/environment adapter.
  official_eval.py Public official-game smoke runner.
  policy.py        Hook decisions: allow, rewrite, block.
  recovery.py      Retry/replan/fallback/abort recovery decisions.
  sandbox.py       Bounded local subprocess sandbox provider.
  checkpoint.py    Latest-step checkpoint persistence.
  context.py       Budgeted context manager and injector.
  delegation.py    SubTask/SubAgent/SubAgentResult and dispatch manager.
  subagents.py     Default perception, diff, and exploration specialists.
  submission.py    Kaggle choose_action/is_done helper entrypoint.
  tracing.py       Trace/span model and JSON persistence.
  validation.py    Frame/action validators.
  thread.py        Codex-like ArcThread API.
  adapters.py      Helpers for Kaggle-style choose_action/is_done agents.
examples/
  stream_and_hooks.py SDK-like event and hook example.
  toy_grid_game.py Small runnable example environment and agent.
scripts/
  check_kaggle_readiness.py Pre-submission readiness report.
tests/
  test_harness.py  Smoke tests for loop, hooks, memory, and thread resume.
```

## Quick Start

```bash
python3 examples/toy_grid_game.py
python3 -m unittest discover -s tests
```

## Core API

```python
from arc_harness import ArcThread, EpisodeRunner

thread = ArcThread(memory_dir=".arc_memory")
result = thread.run_episode(env, agent, max_steps=64)
print(result.status, result.steps, result.done)
```

Use an explicit run config when experiments need stable settings:

```python
from arc_harness import RunnerConfig

config = RunnerConfig(max_steps=128, loop_window=4, stop_on_loop=True)
result = thread.run_episode(env, agent, config=config)
```

Stream events when you want SDK-like observability:

```python
for event in thread.run_streamed(env, agent, max_steps=64):
    print(event["type"], event["episode_id"])
```

Filter hooks with Claude-style matchers:

```python
from arc_harness import ArcThread, HookMatcher

thread = ArcThread(
    hooks=[HookMatcher(hook=my_hook, event="before_action", action="ACTION6")]
)
```

Attach deterministic guardrails:

```python
from arc_harness import ArcThread, CoordinateBoundsGuardrail

thread = ArcThread(guardrails=[CoordinateBoundsGuardrail()])
```

Build compact model-visible context:

```python
from arc_harness import ContextBudget, ContextManager

bundle = thread.build_context(
    latest_frame=latest_frame,
    trace_id=result.summary["trace_id"],
    query="changed cells reward target",
    manager=ContextManager(ContextBudget(max_tokens=1200)),
)
prompt_context = bundle.render()
```

Search durable memory by namespace/category/tags:

```python
thread.memory.add_fact(
    "ACTION6 click near target changed the blue object.",
    category="insight",
    namespace=("action-effects",),
    tags=("action-effect", "action6"),
)

hits = thread.memory.search_entries(
    "blue target click",
    namespace=("action-effects",),
    category="insight",
)
```

For ARC-AGI-3 submissions, implement the Kaggle-facing functions with an agent:

```python
from arc_harness.adapters import KaggleAgentAdapter

adapter = KaggleAgentAdapter(agent)

def is_done(frames, latest_frame):
    return adapter.is_done(frames, latest_frame)

def choose_action(frames, latest_frame):
    return adapter.choose_action(frames, latest_frame)
```

For a Notebook with the package copied in, `arc_harness.submission` exposes the
same functions directly. Configure the default agent with environment variables:

```python
from arc_harness.submission import choose_action, is_done

# Optional:
# ARC_HARNESS_AGENT=delegating_planner
# ARC_HARNESS_AGENT=json_policy
# ARC_HARNESS_POLICY=/kaggle/input/my-policy/policy.json
```

Build a Kaggle-ready folder:

```bash
python3 scripts/build_kaggle_package.py /tmp/arc-harness-kaggle
```

The folder contains `arc_harness/`, a top-level `submission.py`, helper scripts,
and `manifest.json`.

Check the copied package before submitting:

```bash
python3 scripts/check_kaggle_readiness.py \
  --environment-files /kaggle/input/arc-prize-2026-arc-agi-3/environment_files \
  --model-config /kaggle/input/my-policy/model.json
```

The same report is available from Python:

```python
from arc_harness import check_kaggle_readiness

report = check_kaggle_readiness(
    environments_dir="/kaggle/input/arc-prize-2026-arc-agi-3/environment_files",
    model_config="/kaggle/input/my-policy/model.json",
)
print(report.to_dict())
```

## Official ARC-AGI-3 Adapter

The official Kaggle bundle provides `arc_agi.Arcade()` and public
`environment_files`. The harness keeps this dependency optional so unit tests
and Kaggle notebooks remain offline-friendly.

Wrap an already-created official environment:

```python
from arc_harness import OfficialArcEnvironment

env = OfficialArcEnvironment(official_env, game_id="ls20")
result = thread.run_episode(env, agent)
```

Or create one through the official toolkit when `arc_agi` is installed:

```python
from arc_harness import ArcAgi3Config, create_official_environment

env = create_official_environment(
    ArcAgi3Config(
        game_id="ls20",
        operation_mode="OFFLINE",
        environments_dir="/kaggle/input/arc-prize-2026-arc-agi-3/environment_files",
    )
)
```

For `ACTION6`, the adapter sends official action data as `{"x": x, "y": y}`.
It also normalizes official frame objects/dicts into harness `Frame(grid,
status, raw)`.

List public game metadata from a downloaded Kaggle bundle:

```python
from arc_harness import EnvironmentFileCatalog

games = EnvironmentFileCatalog("environment_files").list_games()
```

Run a bounded smoke pass across discovered public games:

```python
from arc_harness import DelegatingPlannerAgent, OfficialSmokeRunner, RunnerConfig

report = OfficialSmokeRunner("environment_files").run(
    DelegatingPlannerAgent(),
    max_games=5,
    config=RunnerConfig(max_steps=32, abort_on_error=False),
)
print(report.to_dict())
```

## Offline Model Integration

Kaggle evaluation disables internet access, so model integrations are local
protocols rather than online API clients.

```python
from arc_harness import CallableModel, ModelBackedAgent, ModelOutput

model = CallableModel(
    lambda model_input: ModelOutput(
        plan=[{"action": {"kind": "ACTION6", "xy": (1, 0)}, "score": 1.0}],
        confidence=0.9,
        rationale="policy file selected a known target",
    ),
    name="LocalPolicy",
)

agent = ModelBackedAgent(model)
```

`JsonPolicyModel` is a tiny deterministic policy useful for smoke tests and
offline notebooks. Larger local models can implement the `LocalModel` protocol:

```python
class MyLocalModel:
    name = "MyLocalModel"

    def predict(self, model_input):
        return ModelOutput(action=("ACTION1"))
```

You can also load a model from a JSON config:

```json
{"type": "json_policy", "path": "/kaggle/input/my-policy/policy.json"}
```

```python
from arc_harness import build_agent_from_model_config

agent = build_agent_from_model_config("/kaggle/input/my-policy/model.json")
```

## Design Notes

The harness separates durable memory from per-episode state.

- `WorkingMemory`: current game trajectory, action effects, loop detection.
- `DurableMemory`: human-readable Markdown facts, procedures, failures, and JSONL episode records.
- `MemoryManager`: search and write facade used by agents and hooks.

Hooks do not decide actions. They observe and enforce:

- `before_episode`
- `after_observe`
- `before_action`
- `after_action`
- `after_episode`
- `on_error`

Hooks can return a `HookDecision`:

- `allow(action)`
- `rewrite(action, reason)`
- `block(reason)`

That keeps the decision policy portable while giving you Claude Code SDK-like
permission and instrumentation points. You can replace a heuristic agent with a
local model-backed agent without rewriting logging, replay, or memory.

## Capability Alignment

| Source | Borrowed capability | Harness implementation |
|---|---|---|
| Claude Code SDK | explicit agent loop | `EpisodeRunner.run_events()` + `StagePipeline` |
| Claude Code SDK | hooks / permissions | `HookManager`, `HookDecision`, `ActionBudgetHook` |
| Claude Code SDK | hook matchers | `HookMatcher(event=..., action=..., status=...)` |
| OpenAI Agents SDK | event stream and tracing | `AgentEvent`, `Trace`, `Span`, `TraceStore` |
| OpenAI Agents SDK | local vs model-visible context | `ContextManager`, `ContextBundle` |
| Codex SDK | start/resume thread shape | `ArcThread`, `ArcThread.resume()` |
| Codex SDK | inspect session state | `list_threads()`, `read_thread()`, metadata |
| Hermes/pi-hermes | layered memory | working memory + durable Markdown/JSONL |
| Hermes/pi-hermes | failure and procedure memory | `FAILURES.md`, `save_skill()` |
| Hermes/pi-hermes | search-first memory policy | `MemoryManager.search()` |
| LangGraph Store | namespace/key long-term memory | `StructuredMemoryStore(namespace=...)` |
| LangGraph semantic search | meaning-based retrieval | lightweight hash-vector hybrid search |
| Pi Hermes Memory | categorized failure/correction/insight memories | `MemoryEntry.category`, SQLite index |
| OpenAI Sessions | persistent session memory | `ArcThread` + `DurableMemory` |
| LangGraph context trimming | trim/summarize state before model calls | budgeted sections and approximate token counting |
| Deep Agents context engineering | offload/compact long-running task context | compact memory/recent-step/trace summaries |
| Claude Code compaction | stable instructions re-injected after compaction | `MemoryPolicy` policy section |
| DeepSeek Harness | capability seams / providers | `CapabilityRegistry`, `ProviderDescriptor` |
| DeepSeek Harness | sandbox as pluggable backend | `LocalSubprocessSandbox` capability provider |

## Stage-Based Agent Loop

`EpisodeRunner` now drives each step through an explicit `StagePipeline` instead
of a monolithic action loop. The default order is:

```text
done_check -> decision -> permission -> action.execute -> stop_check
```

Each stage receives a mutable `LoopState` and a `LoopRuntime` with access to the
environment, agent, memory, hooks, guardrails, trace, checkpoint store, and
event emitter. Stages emit `stage.started`, `stage.completed`, and
`stage.failed` lifecycle events, while preserving domain events such as
`action.proposed`, `action.rewritten`, `action.completed`, and `loop.detected`.

Custom stages can be inserted without changing the runner:

```python
from arc_harness import LoopState, LoopRuntime, StagePipeline

class ReflectStage:
    name = "reflect"

    def run(self, state: LoopState, runtime: LoopRuntime) -> LoopState:
        runtime.memory.add_note(f"reflect before step {state.step}")
        return state

pipeline = StagePipeline()
pipeline.insert_before("decision", ReflectStage())
result = thread.run_episode(env, agent, pipeline=pipeline)
```

This is the seam for planner, subagent, sandbox, recovery, and context-building
stages to participate in the loop directly.

Recovery is also part of the pipeline. When a stage fails, `StagePipeline`
emits `stage.failed`, asks a `RecoveryPolicy` for a decision, and then can retry
the stage, clear the plan for replanning, inject a fallback action, abort the
episode, or re-raise the error. The default policy recovers from invalid action
selection with an untried fallback action, retries perception/exploration/
planning stages once, and keeps guardrail failures as hard errors.

ARC-style delegation can now be expressed as stages instead of being hidden
inside one agent method:

```python
from arc_harness import StagePipeline, delegating_planner_loop_stages

pipeline = StagePipeline(delegating_planner_loop_stages())
result = thread.run_episode(env, agent, pipeline=pipeline)
```

That pipeline runs:

```text
done_check -> perception -> exploration -> planning -> decision -> permission -> action.execute -> stop_check
```

The perception/exploration/planning stages call the registered
`DelegationManager`, save subagent results into `LoopState`, emit
`perception.completed`, `exploration.completed`, and `plan.created`, and then
`PlanDecisionStage` turns the best plan item into the real environment action.

## Capabilities And Sandbox

Runtime providers can be registered behind a common capability/name seam:

```python
from arc_harness import CapabilityRegistry, ProviderDescriptor

registry = CapabilityRegistry()
registry.register(provider, ProviderDescriptor(capability="model", name="local_policy"))
model = registry.require("model", "local_policy")
```

The default registry includes a lightweight local subprocess sandbox:

```python
from arc_harness import DEFAULT_CAPABILITY_REGISTRY, SandboxPolicy

sandbox = DEFAULT_CAPABILITY_REGISTRY.require("sandbox", "local_subprocess")
result = sandbox.run(["python3", "-c", "print('ok')"])
print(result.ok, result.stdout)
```

`LocalSubprocessSandbox` captures stdout/stderr, enforces timeouts, can restrict
commands and working directories, and returns JSON-friendly `SandboxResult`
objects. It is a bounded local process runner, not a container security
boundary; stronger Docker/E2B-style providers should implement the same
`Sandbox` protocol later.

## Context Management

`ContextManager` builds a compact, auditable context bundle from:

- relevant durable memory from `MemoryManager.search_entries()`;
- recent action effects from `WorkingMemory.steps`;
- latest frame summary;
- trace/span summary from `TraceStore`;
- active hypotheses, notes, and failures;
- a compact memory policy section.

Each context source becomes a `ContextSection` with role, title, priority,
metadata, and approximate token count. Sections are trimmed individually and
then fitted into `ContextBudget.max_tokens`; lower-priority sections are dropped
first. This keeps prompt-visible context deterministic and testable while
leaving full replay, memory, and traces on disk.

## Memory Design

The memory layer follows a lightweight version of patterns from LangGraph,
Deep Agents, OpenAI Sessions, and Hermes-style coding-agent memory:

- `WorkingMemory`: short-term, episode-scoped frames, steps, failures, notes, and hypotheses.
- `DurableMemory`: Markdown, JSONL, skills, replay, and structured storage rooted under `.arc_memory/memory`.
- `StructuredMemoryStore`: SQLite-backed long-term memory with namespace, category, tags, metadata, importance, confidence, and source episode id.
- `MemoryPolicy`: policy-only guidance that tells an agent when to retrieve memory instead of injecting all durable memory into context.
- Episode consolidation: every finished episode creates searchable `episode`, `rule`, `failure`, and `insight` memories from summaries and action effects.

Search modes:

- `keyword`: exact token overlap / FTS when available.
- `vector`: lightweight local hashing-vector similarity.
- `hybrid`: combines keyword, vector, importance, and confidence.

## Subagent / Delegation

The delegation layer follows the manager-style pattern used by modern agent
harnesses:

| Reference system | Pattern | Design lesson used here |
|---|---|---|
| OpenAI Agents SDK | agents-as-tools and handoffs | Keep manager control for bounded subtasks; use handoffs only when a specialist should take over. |
| LangGraph / LangChain | supervisor and handoff tools | Pass only task-specific context to specialists and return compact structured outputs. |
| AutoGen | teams and group-chat presets | Specialized agents should have explicit roles, observable outputs, and coordination policy. |
| Claude Code SDK | subagent lifecycle hooks | Subagent start/stop should be traceable and auditable, not hidden inside planner code. |

For ARC, the harness defaults to manager-style delegation. `ArcThread` owns a
`DelegationManager`; the main agent can call specialists without losing control
of the episode:

```python
from arc_harness import ArcThread, DelegationConfig, Frame

thread = ArcThread(memory_dir=".arc_memory")

perception = thread.delegate(
    "perceive",
    {"frame": Frame.from_grid([[1, 1, 0], [0, 2, 2]])},
)

diff = thread.delegate(
    "diff",
    {
        "before": Frame.from_grid([[0, 1], [0, 2]]),
        "after": Frame.from_grid([[0, 3], [0, 2]], status="WIN"),
        "action": ("ACTION6", 1, 0),
    },
)

explore = thread.delegate(
    "explore",
    {
        "frame": Frame.from_grid([[0, 1], [0, 2]]),
        "tried_actions": [("ACTION6", 0, 0), "ACTION1"],
    },
    budget=8,
)

plan = thread.delegate(
    "plan",
    {
        "frame": Frame.from_grid([[0, 1], [0, 2]]),
        "perception": perception.output,
        "candidate_actions": explore.output["competition_values"],
    },
    budget=4,
)
```

Built-in specialists:

- `PerceptionSubAgent`: summarizes colors, grid size, and connected components.
- `DiffSubAgent`: summarizes changed cells, affected bounding box, and status changes.
- `ExplorerSubAgent`: proposes untried coordinate and button actions.
- `PlannerSubAgent`: ranks candidate actions into a compact action plan.

Each delegation returns a `SubAgentResult` with `output`, `summary`,
`confidence`, and lightweight timing trace. Results are also stored under
`namespace=("delegation", kind)` so later episodes can retrieve useful
subagent observations from durable memory.

Delegation calls emit lifecycle events:

- `subtask.started`
- `subtask.retrying`
- `subtask.completed`
- `subtask.failed`

Use `DelegationConfig` for retries, event tracing, memory persistence, and
parallel worker limits:

```python
results = thread.delegate_many(
    [
        ("perceive", {"frame": latest_frame}),
        ("explore", {"frame": latest_frame, "tried_actions": tried_actions}),
    ],
    config=DelegationConfig(max_retries=1, parallel_workers=2),
)

events = thread.read_delegation_events()
trace = thread.read_delegation_trace()
```

Use `DelegatingPlannerAgent` when you want the action policy itself to close the
loop over subagents:

```python
from arc_harness import DelegatingPlannerAgent

agent = DelegatingPlannerAgent()
result = thread.run_episode(env, agent)
```

Use `HandoffAgent` when a specialist should take over action selection after a
state predicate matches:

```python
from arc_harness import HandoffAgent, HandoffRule, HeuristicAgent

agent = HandoffAgent(
    primary=HeuristicAgent(),
    specialists={"explorer": explorer_agent},
    rules=[
        HandoffRule(
            target="explorer",
            reason="switch after repeated misses",
            predicate=lambda state, memory: state["step"] >= 8,
        )
    ],
)
```

## v0.4 Scope

This release is a harness foundation. It intentionally includes:

- a stable action/frame/result data model;
- environment protocol validation;
- direct official ARC-AGI-3 toolkit wrapping through `OfficialArcEnvironment`;
- public `environment_files` metadata discovery;
- official public-game smoke runner via `OfficialSmokeRunner`;
- configurable episode running via `RunnerConfig`;
- structured event streaming;
- stage-based agent loop with insert/replace pipeline stages;
- recovery policy for retry/replan/fallback/abort after stage failures;
- planner stages for perception -> exploration -> planning -> decision;
- action permission hooks;
- hook matchers for event/action/status filtering;
- guardrails for action/frame/result checks;
- trace/span persistence for episode observability;
- budgeted context construction and injection;
- manager-style subagent delegation;
- deterministic perception/diff/exploration/planning specialists;
- subagent lifecycle events;
- retry policy for delegated subtasks;
- parallel `delegate_many()` dispatch;
- delegation trace/span persistence via `read_delegation_trace()`;
- `DelegatingPlannerAgent` for perception -> exploration -> planning action loops;
- `HandoffAgent` and `HandoffRule` for specialist takeover inside an episode;
- thread/session persistence;
- durable memory and typed replay loading;
- SQLite-backed structured memory with namespace/category/tags;
- hybrid keyword/vector memory search without external services;
- automatic episode-to-memory consolidation;
- policy-only memory guidance;
- unified capability registry for model/env/subagent/sandbox providers;
- bounded local subprocess sandbox with timeout and command/cwd policy;
- Kaggle-safe offline model protocol and `ModelBackedAgent`;
- JSON model config loading and Kaggle `submission.py` helpers;
- fail-fast validation for frames/actions;
- structured error context with traceback;
- per-episode checkpoints;
- batch evaluation summaries;
- tiny toy environments and smoke tests.

It intentionally does not yet include:

- learned object perception beyond deterministic connected components;
- a learned planner or search policy;
- local LLM/VLM integration;
- visual replay tooling;
- leaderboard-grade ARC solving behavior.

## Recommended ARC-AGI-3 Extension Path

1. Run `OfficialArcEnvironment` against the public Kaggle `environment_files`
   and collect traces for each game.
2. Extend `RuleLearningAgent` with object extraction and frame-diff features.
3. Replace `JsonPolicyModel`/`CallableModel` with a real local model or search
   policy that implements `LocalModel`.
4. Add action-sequence search on top of `PlannerSubAgent`.
5. Store successful procedures with `DurableMemory.save_skill()`.
6. Use `ArcThread.run_streamed()` plus `JsonlTraceHook` to compare runs.
7. Copy only the needed Python files into Kaggle when preparing the final
   Notebook. Do not depend on online Claude/OpenAI/pi-agent runtimes there.
