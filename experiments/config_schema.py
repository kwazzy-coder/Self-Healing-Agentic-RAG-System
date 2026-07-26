"""
experiments/config_schema.py — Experiment Configuration Schema

Pydantic models for defining reproducible experiment configurations.
Each experiment is defined by a YAML file that specifies all parameters.
"""
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional

class RetrievalConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 7
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    fusion_method: str = "linear"  # "linear" or "rrf"
    enabled: bool = True

class HealingConfig(BaseModel):
    enabled: bool = True
    max_correction_attempts: int = 2
    confidence_threshold: float = 0.6
    use_nli_grounding: bool = True
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    decompose_questions: bool = True

class LLMConfig(BaseModel):
    backend: str = "gemini"
    model_name: Optional[str] = None  # auto-detected from backend
    temperature: float = 0.0
    max_answer_words: int = 8
    context_char_limit: int = 3000

class DatasetConfig(BaseModel):
    name: str = "hotpotqa"  # hotpotqa, ragbench, nq, triviaqa
    split: str = "validation"
    n_samples: int = 100
    subset: Optional[str] = None  # for ragbench: covidqa, msmarco, etc.

class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    seeds: List[int] = [42, 123, 456]
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    healing: HealingConfig = Field(default_factory=HealingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    metrics: List[str] = ["em", "f1", "rouge_l", "faithfulness"]
    compare_baseline: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
