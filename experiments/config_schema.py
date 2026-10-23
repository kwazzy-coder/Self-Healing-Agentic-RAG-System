"""
experiments/config_schema.py — Validated YAML schema for experiments.

Every field left empty (null) falls back to config.py, so each YAML file only
states what that experiment changes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    enabled: bool = True
    embedding_model: Optional[str] = None
    top_k: Optional[int] = None
    bm25_weight: Optional[float] = None
    dense_weight: Optional[float] = None
    fusion_method: Optional[str] = None  # linear | rrf | bm25 | dense


class HealingConfig(BaseModel):
    enabled: bool = True
    max_correction_attempts: Optional[int] = None
    monitor_mode: str = "tiered"  # tiered | heuristic | nli | heuristic_nli | heuristic_llm | llm
    use_nli_grounding: bool = True
    use_llm_judge: bool = True
    nli_low: Optional[float] = None
    nli_high: Optional[float] = None
    decompose_questions: bool = True
    always_correct: bool = False


class LLMConfig(BaseModel):
    backend: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None


class DatasetConfig(BaseModel):
    name: str = "hotpotqa"  # hotpotqa | musique
    n_samples: int = 100
    subset: Optional[str] = None
    question_type: Optional[str] = None


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    seeds: List[int] = Field(default_factory=lambda: [42])
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    healing: HealingConfig = Field(default_factory=HealingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    compare_baseline: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path, encoding="utf-8") as fh:
            return cls(**(yaml.safe_load(fh) or {}))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)

    def config_overrides(self) -> Dict[str, Any]:
        """Global config.py values this experiment changes."""
        o: Dict[str, Any] = {}
        r, h, llm = self.retrieval, self.healing, self.llm
        if r.embedding_model:
            o["EMBEDDING_MODEL"] = r.embedding_model
        if r.bm25_weight is not None:
            o["BM25_WEIGHT"] = r.bm25_weight
        if r.dense_weight is not None:
            o["DENSE_WEIGHT"] = r.dense_weight
        if h.nli_low is not None:
            o["NLI_LOW"] = h.nli_low
        if h.nli_high is not None:
            o["NLI_HIGH"] = h.nli_high
        o["ENABLE_NLI"] = h.use_nli_grounding
        o["ENABLE_LLM_JUDGE"] = h.use_llm_judge
        o["ENABLE_DECOMPOSITION"] = h.decompose_questions
        o["TEMPERATURE"] = llm.temperature
        if llm.reasoning_effort is not None:
            o["LLM_REASONING_EFFORT"] = llm.reasoning_effort or None
        if llm.model_name:
            o["LLM_MODEL"] = llm.model_name
        return o
