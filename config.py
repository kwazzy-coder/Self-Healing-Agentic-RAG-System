"""
config.py — Central configuration for the Self-Healing Agentic RAG system.

Every value can be overridden with an environment variable (or a `.env` file),
so you never need to edit code to switch models or tune the pipeline.

IMPORTANT: other modules read these values as ``config.X`` *at call time*
(never ``from config import X``), so experiment runners can override them
at runtime and the change actually takes effect.

Provider priority when LLM_BACKEND is unset or its key is missing:
  gemini -> groq -> openai -> ollama -> huggingface (local, no key needed)
"""
from __future__ import annotations

import os
from typing import List, Optional


def _load_dotenv(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")) -> None:
    """Minimal .env loader (avoids an extra dependency). Existing env vars win."""
    if not os.path.exists(path):
        path = ".env"  # fall back to the current working directory
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── LLM provider ──────────────────────────────────────────────────────────────
# One of: gemini | groq | openai | ollama | huggingface
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "gemini").lower()
# Optional explicit model id; otherwise the provider default below is used.
LLM_MODEL: Optional[str] = os.getenv("LLM_MODEL") or None

# API keys
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Defaults verified against provider deprecation pages (Oct 2026):
#   - gemini-2.0-flash-lite was shut down on 2026-06-01
#   - Groq's llama-3.1-8b-instant was shut down on 2026-08-16
PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-20b",
    "openai": "gpt-5-mini",
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
    "huggingface": "google/flan-t5-base",
}

# All API providers are reached through the OpenAI-compatible Chat Completions API.
PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openai": None,  # SDK default
    "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1",
}

# Minimum seconds between calls (client-side rate limiting for free tiers).
PROVIDER_MIN_DELAY = {
    "gemini": 4.5,   # ~15 requests/min on the free tier
    "groq": 2.0,     # ~30 requests/min on the free tier
    "openai": 0.0,
    "ollama": 0.0,
    "huggingface": 0.0,
}

# Reasoning models (Gemini 3.x, gpt-oss, gpt-5) think before answering. "low"
# keeps latency/cost down. Set LLM_REASONING_EFFORT="" to omit the parameter.
LLM_REASONING_EFFORT: Optional[str] = os.getenv("LLM_REASONING_EFFORT", "low") or None

TEMPERATURE: float = _env_float("TEMPERATURE", 0.0)
# Reasoning tokens count toward the output budget, so never go below this.
MIN_OUTPUT_TOKENS: int = _env_int("MIN_OUTPUT_TOKENS", 1024)
REQUEST_TIMEOUT_S: float = _env_float("REQUEST_TIMEOUT_S", 60.0)
MAX_ANSWER_WORDS: int = _env_int("MAX_ANSWER_WORDS", 8)

# Context budget given to the generator (characters). Shared fairly across docs
# instead of a single hard cut, so the last retrieved doc is never silently dropped.
CONTEXT_CHAR_LIMIT: int = _env_int("CONTEXT_CHAR_LIMIT", 12000)
HF_CONTEXT_CHAR_LIMIT: int = _env_int("HF_CONTEXT_CHAR_LIMIT", 1500)

# ── Rate limiting / retries ───────────────────────────────────────────────────
MIN_DELAY_BETWEEN_CALLS: Optional[float] = (
    _env_float("MIN_DELAY_BETWEEN_CALLS", 0.0) if os.getenv("MIN_DELAY_BETWEEN_CALLS") else None
)
MAX_RETRIES: int = _env_int("MAX_RETRIES", 4)
RETRY_BASE_WAIT_S: float = _env_float("RETRY_BASE_WAIT_S", 5.0)
RETRY_MAX_WAIT_S: float = _env_float("RETRY_MAX_WAIT_S", 70.0)

# ── Retrieval ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_RETRIEVAL: int = _env_int("TOP_K_RETRIEVAL", 5)
BM25_WEIGHT: float = _env_float("BM25_WEIGHT", 0.3)
DENSE_WEIGHT: float = _env_float("DENSE_WEIGHT", 0.7)
RETRIEVAL_FUSION_METHOD: str = os.getenv("RETRIEVAL_FUSION_METHOD", "linear")  # linear | rrf
RRF_K: int = _env_int("RRF_K", 60)
MAX_DOCS_PER_QUERY: int = _env_int("MAX_DOCS_PER_QUERY", 8)

# ── Self-monitoring ───────────────────────────────────────────────────────────
MAX_CORRECTION_ATTEMPTS: int = _env_int("MAX_CORRECTION_ATTEMPTS", 2)
NLI_MODEL: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
# Tier-2 decision band on P(entailment):
#   < NLI_LOW  -> confidently ungrounded (fail, no LLM call)
#   > NLI_HIGH -> confidently grounded   (pass, no LLM call)
#   otherwise  -> uncertain, escalate to the Tier-3 LLM judge
NLI_LOW: float = _env_float("NLI_LOW", 0.2)
NLI_HIGH: float = _env_float("NLI_HIGH", 0.7)
NLI_MAX_PREMISE_CHARS: int = _env_int("NLI_MAX_PREMISE_CHARS", 1200)
ENABLE_NLI: bool = _env_bool("ENABLE_NLI", True)
ENABLE_LLM_JUDGE: bool = _env_bool("ENABLE_LLM_JUDGE", True)
ENABLE_DECOMPOSITION: bool = _env_bool("ENABLE_DECOMPOSITION", True)

# ── Datasets ──────────────────────────────────────────────────────────────────
HOTPOTQA_SPLIT: str = "validation"
RAGBENCH_SUBSETS: List[str] = ["covidqa", "hotpotqa", "msmarco", "pubmedqa", "expertqa"]
DEFAULT_N_SAMPLES: int = 100
RANDOM_SEED: int = _env_int("RANDOM_SEED", 42)
EXPERIMENT_SEEDS: List[int] = [42, 123, 456]

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "./results")


def default_model_for(provider: str) -> str:
    """Model id to use for a provider (LLM_MODEL overrides the default)."""
    return LLM_MODEL or PROVIDER_DEFAULT_MODELS.get(provider, "")


def min_delay_for(provider: str) -> float:
    """Client-side delay between API calls for a provider."""
    if MIN_DELAY_BETWEEN_CALLS is not None:
        return MIN_DELAY_BETWEEN_CALLS
    return PROVIDER_MIN_DELAY.get(provider, 0.0)
