"""Kaggle submission helpers.

This module is intentionally import-light. A Notebook can copy the package and
then expose the two competition functions by delegating to this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from arc_harness.integrations.adapters import KaggleAgentAdapter
from arc_harness.core.agent import ArcAgent, DelegatingPlannerAgent, HeuristicAgent
from arc_harness.models.models import CandidateRankingAgent, JsonPolicyModel, ModelBackedAgent, QwenLocalRanker, build_agent_from_model_config


def build_agent_from_env() -> ArcAgent:
    """Create a Kaggle-safe agent from environment variables.

    Supported values:
    - `ARC_HARNESS_AGENT=delegating_planner`
    - `ARC_HARNESS_AGENT=json_policy` with `ARC_HARNESS_POLICY=/path/policy.json`
    - `ARC_HARNESS_AGENT=model_config` with `ARC_HARNESS_MODEL_CONFIG=/path/model.json`
    - `ARC_HARNESS_AGENT=qwen_ranker` with `ARC_HARNESS_MODEL_PATH=/kaggle/input/...`
    - anything else falls back to `HeuristicAgent`
    """

    kind = os.environ.get("ARC_HARNESS_AGENT", "delegating_planner").strip().lower()
    if kind in {"delegating_planner", "planner", "default"}:
        return DelegatingPlannerAgent()
    if kind in {"json_policy", "policy"}:
        policy_path = os.environ.get("ARC_HARNESS_POLICY")
        if not policy_path:
            raise ValueError("ARC_HARNESS_POLICY must be set when ARC_HARNESS_AGENT=json_policy.")
        return ModelBackedAgent(JsonPolicyModel(policy_path))
    if kind in {"model_config", "model"}:
        config_path = os.environ.get("ARC_HARNESS_MODEL_CONFIG")
        if not config_path:
            raise ValueError("ARC_HARNESS_MODEL_CONFIG must be set when ARC_HARNESS_AGENT=model_config.")
        return build_agent_from_model_config(config_path)
    if kind in {"qwen_ranker", "qwen", "local_qwen"}:
        model_path = os.environ.get("ARC_HARNESS_MODEL_PATH")
        if not model_path:
            raise ValueError("ARC_HARNESS_MODEL_PATH must be set when ARC_HARNESS_AGENT=qwen_ranker.")
        return CandidateRankingAgent(QwenLocalRanker(model_path))
    if kind in {"heuristic", "baseline"}:
        return HeuristicAgent()
    raise ValueError(f"Unknown ARC_HARNESS_AGENT={kind!r}.")


def build_adapter(agent: ArcAgent | None = None, memory_dir: str | Path | None = None) -> KaggleAgentAdapter:
    root = memory_dir or os.environ.get("ARC_HARNESS_MEMORY_DIR", "/tmp/arc_harness_memory")
    return KaggleAgentAdapter(agent or build_agent_from_env(), memory_dir=root)


_ADAPTER = build_adapter()


def is_done(frames: Sequence[Any], latest_frame: Any) -> bool:
    return _ADAPTER.is_done(frames, latest_frame)


def choose_action(frames: Sequence[Any], latest_frame: Any) -> Any:
    return _ADAPTER.choose_action(frames, latest_frame)
