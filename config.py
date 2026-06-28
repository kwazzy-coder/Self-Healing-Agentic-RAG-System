"""
config.py — Central configuration.

Backend priority (auto-detected):
  1. "gemini"      — Gemini 2.5 Flash-Lite (FREE, best quality, recommended)
  2. "groq"        — Llama 3.1 8B (FREE, high volume)
  3. "ollama"      — Local Llama (FREE, no internet)
  4. "openai"      — GPT-3.5/4 (paid)
  5. "huggingface" — Flan-T5 (free but weak)
"""
import os

# ── LLM Backend ───────────────────────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")

# Gemini settings (FREE — get key at aistudio.google.com/apikey)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash-lite"   # 1000 RPD, 15 RPM, free

# Groq settings (FREE fallback — console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"      # 14400 RPD, 30 RPM, free

# Ollama settings (local fallback)
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = "http://localhost:11434"

# OpenAI settings (paid)
OPENAI_MODEL   = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# HuggingFace last resort
HF_MODEL = "google/flan-t5-base"

# ── Rate Limiting ─────────────────────────────────────────────────────────────
# Gemini 2.0 Flash-Lite: 15 RPM → 1 per 4 seconds
# Groq llama-3.1-8b:     30 RPM → 1 per 2 seconds
MIN_DELAY_BETWEEN_CALLS = 4.5   # safe for Gemini (15 RPM)
MAX_RETRIES             = 3
RETRY_WAIT_SECONDS      = 65    # wait just over 1 min to reset RPM window

# ── Retrieval ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 7
BM25_WEIGHT     = 0.3
DENSE_WEIGHT    = 0.7

# ── Self-Monitoring ───────────────────────────────────────────────────────────
MAX_CORRECTION_ATTEMPTS = 2

# ── Dataset ───────────────────────────────────────────────────────────────────
HOTPOTQA_SPLIT    = "validation"
RAGBENCH_SUBSETS  = ["covidqa", "hotpotqa", "msmarco", "pubmedqa", "expertqa"]
DEFAULT_N_SAMPLES = 100
RANDOM_SEED       = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR   = "./cache"
RESULTS_DIR = "./results"
