"""
baselines/ — Comparison systems that share the same models as the full system.

Every system exposes ``run(question, documents, question_type=None) -> PipelineResult``,
so the evaluator can treat them interchangeably.

| name            | planning | monitor        | correction            | what it isolates            |
|-----------------|----------|----------------|-----------------------|-----------------------------|
| vanilla_rag     | no       | -              | -                     | plain retrieve + generate   |
| planner_only    | yes      | -              | -                     | value of decomposition      |
| self_healing    | yes      | tiered (1-2-3) | adaptive, up to 2     | full system                 |
| nli_only        | yes      | tiers 1-2      | adaptive              | value of the LLM judge      |
| llm_judge_only  | yes      | LLM every time | adaptive              | cost of skipping cheap tiers|
| always_correct  | yes      | none           | 1 round, every query  | value of selective healing  |
| crag            | no       | doc grading    | re-retrieve if bad    | CRAG-style prior work       |
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from pipeline import Components, PipelineConfig, SelfHealingRAGPipeline

SYSTEM_CONFIGS: Dict[str, PipelineConfig] = {
    "vanilla_rag": PipelineConfig(name="vanilla_rag", enable_planning=False, enable_healing=False),
    "planner_only": PipelineConfig(name="planner_only", enable_healing=False),
    "self_healing": PipelineConfig(name="self_healing"),
    "nli_only": PipelineConfig(name="nli_only", monitor_mode="heuristic_nli"),
    "llm_judge_only": PipelineConfig(name="llm_judge_only", monitor_mode="llm"),
    "always_correct": PipelineConfig(name="always_correct", always_correct=True),
}
ALL_SYSTEMS = [*SYSTEM_CONFIGS, "crag"]


def build_system(name: str, components: Components, **overrides: Any):
    """Create a runnable system by name, sharing the given model components."""
    if name == "crag":
        from baselines.crag_baseline import CRAGBaseline

        return CRAGBaseline(components=components, short_answers=overrides.get("short_answers", True))
    if name not in SYSTEM_CONFIGS:
        raise ValueError(f"Unknown system '{name}'. Choose from {ALL_SYSTEMS}")
    cfg = replace(SYSTEM_CONFIGS[name], **overrides)
    return SelfHealingRAGPipeline(pipeline_config=cfg, components=components)
