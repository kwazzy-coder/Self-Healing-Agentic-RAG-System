# Design and Evaluation of a Self-Healing Agentic RAG System
### for Robust Multi-Turn Task Execution

---

## 1. Introduction

Large Language Models (LLMs) have transformed natural language understanding, but they remain prone to hallucination—generating confident but factually incorrect outputs. Retrieval-Augmented Generation (RAG) mitigates this by grounding responses in external documents, but existing RAG systems still fail in complex multi-turn and multi-hop scenarios due to:

- Retrieval of irrelevant or incomplete documents
- Error propagation during multi-step reasoning
- No mechanism to detect or correct their own failures

This project implements a **Self-Healing Agentic RAG system** that autonomously detects failures in its reasoning loop and corrects them before delivering a final answer.

---

## 2. Related Work

| Paper | Contribution |
|-------|-------------|
| Lewis et al. (2020) — RAG | Foundational retrieval-augmented generation framework |
| Asai et al. (2023) — Self-RAG | Self-reflection tokens for selective retrieval and critique |
| Yao et al. (2022) — ReAct | Interleaved reasoning and action in LLM agents |
| Shinn et al. (2023) — Reflexion | Verbal reinforcement via self-reflection for task improvement |
| Yang et al. (2018) — HotpotQA | Multi-hop QA benchmark with supporting fact supervision |
| Manakul et al. (2023) — RAGBench | Faithfulness & grounding evaluation framework for RAG |

---

## 3. System Architecture

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
│  LLM Reasoner           │  GPT-3.5-turbo or Flan-T5
│  (llm_reasoner.py)      │  Generates answer from context
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

---

## 4. Datasets

### 4.1 HotpotQA (Primary Benchmark)
- **Source:** https://hotpotqa.github.io/
- **Size:** 7,405 validation samples (distractor setting)
- **Task:** Multi-hop question answering requiring 2 supporting documents
- **Why:** Gold supporting facts available → lets us measure retrieval quality separately from answer quality
- **Settings used:**
  - *Distractor:* 10 candidate paragraphs per question (8 distractors + 2 gold)
  - Question types: `bridge` (sequential) and `comparison` (parallel)

**Metrics on HotpotQA:**
| Metric | Formula |
|--------|---------|
| Answer EM | Exact match after normalization |
| Answer F1 | Token-level precision/recall F1 |
| Supporting Facts F1 | F1 of retrieved titles vs gold supporting titles |
| Joint EM / F1 | Both answer and supporting facts correct |

### 4.2 RAGBench (Faithfulness Benchmark)
- **Source:** https://huggingface.co/datasets/galileo-ai/ragbench
- **Size:** 12 subsets, ~96k samples total
- **Subsets used:** `covidqa`, `hotpotqa`, `msmarco`, `pubmedqa`, `expertqa`
- **Why:** Provides per-sentence grounding labels and multiple faithfulness scores from different methods (RAGAS, TruLens, GPT-4 judge)
- **Key fields used:**
  - `adherence_score` (bool): Is the response fully grounded?
  - `ragas_faithfulness`: RAGAS faithfulness score [0,1]
  - `trulens_groundedness`: TruLens grounding score [0,1]
  - `sentence_support_information`: Per-sentence support labels
  - `unsupported_response_sentence_keys`: Which sentences are hallucinated

**Metrics on RAGBench:**
| Metric | Description |
|--------|-------------|
| Hallucination Detection Accuracy | Can self-monitor correctly flag hallucinated samples? |
| System Grounded Rate | % of outputs where no failure is flagged |
| F1 vs Gold Response | Token F1 against the GPT-3.5 baseline answer |

---

## 5. Self-Healing Mechanism (Technical Detail)

### 5.1 Self-Monitoring Checks

**Check 1 — Confidence (Heuristic):**
Scans the generated answer for uncertainty phrases:
> "I don't know", "not mentioned", "cannot be determined", etc.

If found → `confidence_ok = False` → trigger corrective action.

**Check 2 — Grounding (LLM NLI):**
Splits the answer into sentences. Prompts the LLM:
> "Which of these sentences are NOT supported by the retrieved documents?"

Any unsupported sentence → `grounding_ok = False`.

**Check 3 — Completeness:**
Checks if the answer is non-empty and doesn't start with an uncertainty phrase.

**Check 4 — Consistency (Multi-hop):**
For multi-hop questions, prompts the LLM:
> "Do these sub-answers contradict each other? YES or NO."

### 5.2 Corrective Actions

| Action | Trigger | Method |
|--------|---------|--------|
| `reformulate` | Low confidence | LLM rephrases query → re-retrieve → re-generate |
| `re_retrieve` | Ungrounded sentences | LLM generates keyword expansion → combined retrieval → re-generate |
| `decompose` | Incomplete answer | LLM decomposes into 2 sub-Qs → answer each → synthesize |

Max 2 correction rounds per query (configurable via `MAX_CORRECTION_ATTEMPTS`).

---

## 6. Experimental Setup

### 6.1 Models
| Component | Choice | Alternative |
|-----------|--------|------------|
| Embeddings | `all-MiniLM-L6-v2` | any sentence-transformer |
| Retrieval | BM25 (30%) + FAISS (70%) | dense-only |
| LLM | GPT-3.5-turbo | Flan-T5-base (free) |
| Monitoring | LLM-based NLI | keyword heuristic |

### 6.2 Hyperparameters
```python
TOP_K_RETRIEVAL = 5
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7
CONFIDENCE_THRESHOLD = 0.6
MAX_CORRECTION_ATTEMPTS = 2
```

### 6.3 Evaluation Protocol
- Sample 100–200 examples from each dataset split
- Report mean EM, F1, hallucination rate, correction rate
- Compare Self-Healing RAG vs Baseline RAG (no self-healing)

---

## 7. Expected Results

Based on prior work (Self-RAG, CRAG), we expect:

| System | EM | F1 | Halluc. Rate | Correction Rate |
|--------|----|----|-------------|----------------|
| Baseline RAG | ~45% | ~58% | ~25% | 0% |
| Self-Healing RAG | ~52% | ~65% | ~12% | ~60% |

The self-healing system should show:
- **+5–10% F1** on HotpotQA (especially bridge questions)
- **~50% reduction in hallucination rate** on RAGBench covidqa/pubmedqa
- **Correction Rate ~60%**: 60% of detected failures successfully corrected

---

## 8. How to Run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key (skip if using HuggingFace)
export OPENAI_API_KEY="sk-..."

# 3. Quick demo (no API key needed)
python demo.py

# 4. HotpotQA evaluation
python run_evaluation.py --dataset hotpotqa --n_samples 100 --compare_baseline

# 5. RAGBench evaluation
python run_evaluation.py --dataset ragbench --subset covidqa --n_samples 100
python run_evaluation.py --dataset ragbench --subset pubmedqa --n_samples 100

# 6. Full evaluation (both datasets)
python run_evaluation.py --dataset both --n_samples 100
```

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestrator — entry point for all runs |
| `agent/task_planner.py` | Multi-hop decomposition agent |
| `agent/llm_reasoner.py` | LLM interface (OpenAI / HuggingFace) |
| `agent/self_monitor.py` | **Self-healing core** — failure detection |
| `agent/corrective_actions.py` | Corrective strategies |
| `retrieval/retriever.py` | Hybrid BM25 + FAISS retrieval |
| `retrieval/vector_store.py` | FAISS index management |
| `data/hotpotqa_loader.py` | HotpotQA dataset loading |
| `data/ragbench_loader.py` | RAGBench dataset loading |
| `evaluation/metrics.py` | EM, F1, faithfulness, hallucination rate |
| `evaluation/evaluator.py` | End-to-end evaluation runner |
| `run_evaluation.py` | CLI entry point |
| `demo.py` | Quick 3-sample smoke test |
| `config.py` | All hyperparameters |

---

## 10. References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
2. Asai, A., et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024.
3. Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
4. Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
5. Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP.
6. Saad-Falcon, J., et al. (2023). *RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems*. arXiv.
