"""
config.py — Central configuration for Self-Healing Agentic RAG System.

Backend priority (auto-detected):
  1. "gemini"      — Gemini 2.5 Flash-Lite (FREE, best quality, recommended)
  2. "groq"        — Llama 3.1 8B (FREE, high volume)
  3. "ollama"      — Local Llama (FREE, no internet)
  4. "openai"      — GPT-3.5/4 (paid)
  5. "huggingface" — Flan-T5 (free but weak)

This module defines all configuration settings for reproducing the paper's experiments.
"""
import os
from dataclasses import dataclass
from typing import List

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── LLM Backend ───────────────────────────────────────────────────────────────
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "gemini")

# Gemini settings (FREE — get key at aistudio.google.com/apikey)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.0-flash-lite"   # 1000 RPD, 15 RPM, free

# Groq settings (FREE fallback — console.groq.com)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.1-8b-instant"      # 14400 RPD, 30 RPM, free

# Ollama settings (local fallback)
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL: str = "http://localhost:11434"

# OpenAI settings (paid)
OPENAI_MODEL: str = "gpt-3.5-turbo"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# HuggingFace last resort
HF_MODEL: str = "google/flan-t5-base"

# Settings for LLM generation
TEMPERATURE: float = 0.0
MAX_ANSWER_WORDS: int = 8
CONTEXT_CHAR_LIMIT: int = 3000
HF_CONTEXT_CHAR_LIMIT: int = 600

# ── Rate Limiting ─────────────────────────────────────────────────────────────
MIN_DELAY_BETWEEN_CALLS: float = 4.5   # safe for Gemini (15 RPM)
MAX_RETRIES: int = 3
RETRY_WAIT_SECONDS: int = 65    # wait just over 1 min to reset RPM window

# ── Retrieval ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL: int = 7
BM25_WEIGHT: float = 0.3
DENSE_WEIGHT: float = 0.7
RETRIEVAL_FUSION_METHOD: str = "linear"  # Alternatives: "rrf"
MAX_DOCS_PER_QUERY: int = 10

# ── Self-Monitoring ───────────────────────────────────────────────────────────
MAX_CORRECTION_ATTEMPTS: int = 2
CONFIDENCE_THRESHOLD: float = 0.6
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"

# ── Dataset ───────────────────────────────────────────────────────────────────
HOTPOTQA_SPLIT: str = "validation"
RAGBENCH_SUBSETS: List[str] = ["covidqa", "hotpotqa", "msmarco", "pubmedqa", "expertqa"]
DEFAULT_N_SAMPLES: int = 100
RANDOM_SEED: int = 42
EXPERIMENT_SEEDS: List[int] = [42, 123, 456, 789, 1337]

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR: str = "./cache"
RESULTS_DIR: str = "./results"

# ── Experiment Infrastructure ──────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """Bundles all experiment-relevant settings."""
    llm_backend: str = LLM_BACKEND
    temperature: float = TEMPERATURE
    embedding_model: str = EMBEDDING_MODEL
    top_k: int = TOP_K_RETRIEVAL
    max_correction_attempts: int = MAX_CORRECTION_ATTEMPTS
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    seeds: List[int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = EXPERIMENT_SEEDS
