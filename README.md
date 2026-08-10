# Self-Healing Agentic RAG System

A fully implemented **Self-Healing Agentic Retrieval-Augmented Generation (RAG)** system that autonomously detects failures in its own reasoning pipeline and corrects them before returning a final answer. Evaluated on **HotpotQA** (multi-hop reasoning) and **RAGBench** (faithfulness/grounding scoring).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture](#3-architecture)
4. [Datasets](#4-datasets)
5. [Setup & Installation](#5-setup--installation)
6. [LLM Backend Configuration](#6-llm-backend-configuration)
7. [Running Experiments](#7-running-experiments)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Expected Results](#9-expected-results)
10. [Troubleshooting](#10-troubleshooting)
11. [References](#11-references)

---

## 1. Project Overview

Large Language Models (LLMs) remain prone to hallucination — generating confident but factually incorrect outputs. While Retrieval-Augmented Generation (RAG) mitigates this by grounding responses in external documents, standard RAG systems still fail in complex multi-hop and multi-turn scenarios because:

- Retrieved documents may be irrelevant or incomplete
- Errors propagate through multi-step reasoning chains
- There is no mechanism for the system to detect or correct its own failures

This project implements a **Self-Healing Agentic RAG system** with four core capabilities:

| Capability | Description |
|---|---|
| **Task Planning** | Decomposes multi-hop questions into sub-questions |
| **Hybrid Retrieval** | Combines BM25 (sparse) + FAISS (dense) retrieval |
| **Self-Monitoring** | Detects failures via 4 checks: confidence, grounding, completeness, consistency |
| **Corrective Actions** | Autonomously reformulates queries, re-retrieves documents, or decomposes questions |

---

## 2. Repository Structure

```
self_healing_rag/
│
├── README.md                    ← This file
├── REPORT.md                    ← Full technical report with architecture & results
├── WHAT_TO_DO.md                ← Step-by-step guide to reproduce all experiments
├── requirements.txt             ← Pinned Python dependencies (reproducibility)
├── config.py                    ← Central configuration + ExperimentConfig dataclass
│
├── pipeline.py                  ← Main RAG pipeline orchestrator (start here)
├── run_evaluation.py            ← CLI entry point — runs full evaluation
├── demo.py                      ← Quick 5-sample smoke test
│
├── data/                        ← Dataset loading modules
│   ├── hotpotqa_loader.py       ← HotpotQA (multi-hop, 2-hop)
│   ├── ragbench_loader.py       ← RAGBench subsets (COVID, MSMARCO, PubMed, ...)
│   ├── musique_loader.py        ← MuSiQue (multi-hop, 2-4 hops)
│   ├── nq_loader.py             ← Natural Questions (single-hop)
│   └── popqa_loader.py          ← PopQA (long-tail entities)
│
├── retrieval/                   ← Document retrieval
│   ├── vector_store.py          ← FAISS-based dense vector index (cosine similarity)
│   └── retriever.py             ← Hybrid BM25 + FAISS with normalized linear/RRF fusion
│
├── agent/                       ← Agentic reasoning components
│   ├── task_planner.py          ← Question type detection + multi-hop decomposition
│   ├── llm_reasoner.py          ← Multi-backend LLM with CoT prompting & rate limiting
│   ├── self_monitor.py          ← ★ 3-tier cascading monitor (heuristic → NLI → LLM)
│   ├── corrective_actions.py    ← Failure-type-driven adaptive correction strategies
│   └── cost_tracker.py          ← Per-query cost tracking (LLM calls, NLI, latency)
│
├── evaluation/                  ← Publication-quality evaluation suite
│   ├── metrics.py               ← EM, F1, ROUGE-L, BERTScore, retrieval metrics
│   ├── faithfulness.py          ← RAG Triad: faithfulness, relevancy, context relevancy
│   ├── statistical.py           ← Bootstrap CI, paired significance tests, McNemar
│   ├── evaluator.py             ← End-to-end evaluation with per-type breakdown
│   └── cost_analysis.py         ← Cost-efficiency tables & savings computation
│
├── baselines/                   ← SOTA baseline implementations for comparison
│   ├── vanilla_rag.py           ← Standard RAG (no correction)
│   ├── always_correct.py        ← Always-correct (no selective monitoring)
│   ├── llm_judge_only.py        ← LLM-as-judge on every query (no tiered approach)
│   └── crag_baseline.py         ← CRAG-style document grading + re-retrieval
│
├── experiments/                 ← Experiment infrastructure
│   ├── config_schema.py         ← Pydantic schema for experiment configs
│   ├── runner.py                ← Multi-seed experiment runner with metadata logging
│   ├── run_all_experiments.py   ← Master script for all ablations
│   └── configs/                 ← 12 YAML configs (full system, baselines, ablations)
│
├── analysis/                    ← Paper-ready analysis tools
│   ├── result_analyzer.py       ← LaTeX table generation, system comparison
│   ├── visualizations.py        ← Publication-quality plots (matplotlib/seaborn)
│   ├── error_analysis.py        ← Failure categorization & pattern analysis
│   └── case_studies.py          ← Qualitative case study extraction
│
├── paper/                       ← Paper generation utilities
│   └── algorithm.py             ← LaTeX pseudocode (Algorithm 1 & 2)
│
└── utils/
    └── helpers.py               ← Text processing, logging setup, system info
```

### Key files at a glance

| File | Role |
|---|---|
| `pipeline.py` | **Start here.** Orchestrates all modules for a single query |
| `agent/self_monitor.py` | **Core innovation.** Runs 4 failure-detection checks |
| `agent/corrective_actions.py` | **Core innovation.** Applies self-healing corrections |
| `run_evaluation.py` | CLI entry point for all experiments |
| `config.py` | All hyperparameters and API key settings in one place |
| `demo.py` | Quick sanity check, no API key required |

---

## 3. Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Task Planner Agent    │  Detects bridge/comparison multi-hop
│   (task_planner.py)     │  Decomposes into sub-questions
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Hybrid Retriever       │  BM25 (30%) + FAISS Dense (70%)
│  (retriever.py)         │  Indexes per-sample documents
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  LLM Reasoner           │  Gemini / Groq / OpenAI / HuggingFace
│  (llm_reasoner.py)      │  Generates answer from retrieved context
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  Self-Monitoring Module  (self_monitor.py)       │
│                                                  │
│  Check 1: Confidence    — uncertainty phrases?   │
│  Check 2: Grounding     — LLM NLI per sentence   │
│  Check 3: Completeness  — does answer exist?     │
│  Check 4: Consistency   — sub-answers conflict?  │
└──────────────┬──────────────────────┬────────────┘
               │ PASS                 │ FAIL
               ▼                      ▼
         Final Answer      ┌──────────────────────┐
                           │  Corrective Actions   │
                           │  (corrective_actions) │
                           │                       │
                           │  • Reformulate query  │
                           │  • Re-retrieve docs   │
                           │  • Decompose question │
                           └──────────┬────────────┘
                                      │ (max 2 attempts)
                                      └──► back to Retriever
```

### Self-Monitoring Checks (Detail)

| Check | Method | Failure Signal |
|---|---|---|
| **Confidence** | Scans for uncertainty phrases ("I don't know", "not mentioned", etc.) | Phrase found |
| **Grounding** | LLM-based NLI — asks which sentences are unsupported by retrieved docs | Any unsupported sentence |
| **Completeness** | Answer is non-empty and doesn't begin with an uncertainty phrase | Empty or uncertain |
| **Consistency** | LLM checks whether sub-answers contradict each other (multi-hop only) | Contradiction detected |

### Corrective Actions (Detail)

| Action | Trigger | Method |
|---|---|---|
| `reformulate` | Low confidence | LLM rephrases query → re-retrieve → re-generate |
| `re_retrieve` | Ungrounded sentences | LLM generates keyword expansion → augmented retrieval → re-generate |
| `decompose` | Incomplete answer | LLM decomposes into 2 sub-questions → answer each → synthesize |

Maximum 2 correction rounds per query (configurable via `MAX_CORRECTION_ATTEMPTS` in `config.py`). The pipeline always returns the **best answer seen** across all rounds — if a correction degrades quality, the previous answer is preserved.

---

## 4. Datasets

### HotpotQA (Primary Benchmark)
- **Source:** https://hotpotqa.github.io/
- **Split used:** Validation set (7,405 samples), distractor setting
- **Task:** Multi-hop QA requiring exactly 2 supporting documents from 10 candidates (8 distractors + 2 gold)
- **Question types:** `bridge` (sequential reasoning) and `comparison` (parallel reasoning)
- **Loader:** `data/hotpotqa_loader.py`

### RAGBench (Faithfulness Benchmark)
- **Source:** https://huggingface.co/datasets/galileo-ai/ragbench
- **Subsets used:** `covidqa`, `hotpotqa`, `msmarco`, `pubmedqa`, `expertqa`
- **Task:** Faithfulness evaluation — per-sentence grounding labels and hallucination flags
- **Key fields:** `adherence_score`, `ragas_faithfulness`, `trulens_groundedness`, `unsupported_response_sentence_keys`
- **Loader:** `data/ragbench_loader.py`

---

## 5. Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/self_healing_rag.git
cd self_healing_rag
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **If `faiss-cpu` fails:**
> ```bash
> pip install faiss-cpu --no-cache-dir
> ```
> **If `torch` is slow:**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### Step 4 — Verify the install

```bash
python demo.py
```

Expected output: 3 questions answered with trace logs showing the self-monitoring loop. No API key is required for this step (uses HuggingFace Flan-T5 locally).

---

## 6. LLM Backend Configuration

The system auto-detects the best available backend in this priority order:

| Priority | Backend | Cost | Quality | Setup |
|---|---|---|---|---|
| 1 | **Gemini 2.0 Flash-Lite** | Free | Best | Get key at [aistudio.google.com](https://aistudio.google.com/apikey) |
| 2 | **Groq (Llama 3.1 8B)** | Free | High | Get key at [console.groq.com](https://console.groq.com) |
| 3 | **Ollama (local)** | Free | Medium | See `SETUP_OLLAMA.md` |
| 4 | **OpenAI GPT-3.5** | Paid (~$1–2 / 200 samples) | High | Set `OPENAI_API_KEY` |
| 5 | **HuggingFace Flan-T5** | Free | Low | No setup needed |

### Setting your API key

```bash
# Option A — Gemini (recommended, free)
export GEMINI_API_KEY="your-key-here"
export LLM_BACKEND=gemini

# Option B — Groq (free, high volume)
export GROQ_API_KEY="your-key-here"
export LLM_BACKEND=groq

# Option C — OpenAI (paid)
export OPENAI_API_KEY="sk-your-key-here"
export LLM_BACKEND=openai

# Option D — HuggingFace only (no key needed)
export LLM_BACKEND=huggingface
```

All settings can also be edited directly in `config.py`.

---

## 7. Running Experiments

### Quick demo (no API key needed)

```bash
python demo.py
```

### Experiment 1 — HotpotQA: Self-Healing vs Baseline

```bash
python run_evaluation.py \
    --dataset hotpotqa \
    --n_samples 200 \
    --setting distractor \
    --compare_baseline
```

This produces the **core comparison** — EM, F1, hallucination rate, and correction rate for both the self-healing system and the no-correction baseline.

### Experiment 2 — HotpotQA by question type

```bash
# Bridge questions (sequential multi-hop)
python run_evaluation.py --dataset hotpotqa --n_samples 100 --question_type bridge

# Comparison questions (parallel multi-hop)
python run_evaluation.py --dataset hotpotqa --n_samples 100 --question_type comparison
```

### Experiment 3 — RAGBench hallucination detection

```bash
python run_evaluation.py --dataset ragbench --subset covidqa --n_samples 200
python run_evaluation.py --dataset ragbench --subset hotpotqa --n_samples 200
python run_evaluation.py --dataset ragbench --subset msmarco --n_samples 200
python run_evaluation.py --dataset ragbench --subset pubmedqa --n_samples 200
```

This reports `hallucination_detection_accuracy` — how well the self-monitor identifies hallucinated responses.

### Experiment 4 — Full evaluation (both datasets)

```bash
python run_evaluation.py --dataset both --n_samples 100 --compare_baseline
```

### Results output

All results are saved to the `results/` directory (auto-created):

```
results/
├── hotpotqa_results.json          ← Per-sample results + aggregate metrics
├── ragbench_covidqa_results.json
├── ragbench_hotpotqa_results.json
└── summary.json                   ← Cross-dataset summary table
```

---

## 8. Evaluation Metrics

| Metric | Description | Dataset |
|---|---|---|
| **EM (Exact Match)** | Answer matches gold string exactly after normalization | HotpotQA |
| **F1** | Token-level precision/recall F1 overlap | HotpotQA |
| **Supporting Facts F1** | F1 of retrieved doc titles vs gold supporting titles | HotpotQA |
| **Hallucination Rate** | % of responses containing at least one unsupported claim | Both |
| **Correction Rate** | % of detected failures successfully corrected | Both |
| **Hallucination Detection Accuracy** | How accurately the self-monitor flags hallucinated samples | RAGBench |
| **System Grounded Rate** | % of outputs where no failure is flagged | RAGBench |
| **Task Success Rate** | % of questions answered fully correctly | HotpotQA |

---

## 9. Expected Results

Based on prior work (Self-RAG, CRAG) and architecture design:

| System | EM | F1 | Hallucination Rate | Correction Rate |
|---|---|---|---|---|
| Baseline RAG | ~45% | ~58% | ~25% | 0% |
| **Self-Healing RAG** | **~52%** | **~65%** | **~12%** | **~60%** |

Key improvements expected:
- **+5–10% F1** on HotpotQA (especially bridge-type questions)
- **~50% reduction** in hallucination rate on RAGBench covidqa/pubmedqa
- **Correction Rate ~60%**: 60% of detected failures successfully corrected within 2 rounds

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `ModuleNotFoundError: rank_bm25` | `pip install rank-bm25` |
| `ModuleNotFoundError: datasets` | `pip install datasets` |
| HuggingFace model download is slow | Run once — it caches locally in `~/.cache/huggingface` |
| `OpenAI RateLimitError` | Use `--n_samples 50` or add `time.sleep(1)` in evaluator loop |
| Gemini 429 (rate limit) | `MIN_DELAY_BETWEEN_CALLS` in `config.py` is set to 4.5s; increase if needed |
| `CUDA out of memory` | Set `LLM_BACKEND=openai` or `LLM_BACKEND=gemini` instead of local HF model |
| Ollama not found | Follow `SETUP_OLLAMA.md` to install and start the local server |

---

## 11. References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
2. Asai, A., et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024.
3. Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
4. Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
5. Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP.
6. Saad-Falcon, J., et al. (2023). *RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems*. arXiv.
