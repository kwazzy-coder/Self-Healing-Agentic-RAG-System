# What To Do — Complete Step-by-Step Guide
### Self-Healing Agentic RAG System
#### Team: PES1UG23CS506, PES1UG23CS272, PES1UG23CS252

---

## Phase 1 — Setup (Day 1)

### 1.1 Create a Python virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 1.2 Install dependencies
```bash
pip install -r requirements.txt
```
> If `faiss-cpu` fails: `pip install faiss-cpu --no-cache-dir`
> If torch is slow to install: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 1.3 Set your LLM backend

**Option A — Free (no API key needed, slower):**
```bash
export LLM_BACKEND=huggingface
```

**Option B — OpenAI (faster, better results, costs ~$1-2 for 200 samples):**
```bash
export OPENAI_API_KEY=sk-your-key-here
export LLM_BACKEND=openai
```

---

## Phase 2 — Verify Everything Works (Day 1)

### 2.1 Run the demo (3 samples, no API key needed)
```bash
python demo.py
```
Expected output: 3 questions answered, with trace logs showing the self-monitoring loop.

---

## Phase 3 — Run Experiments (Days 2–3)

### Experiment 1: HotpotQA — Self-Healing vs Baseline
```bash
python run_evaluation.py \
    --dataset hotpotqa \
    --n_samples 200 \
    --setting distractor \
    --compare_baseline
```
**This gives you:** EM, F1, hallucination rate, correction rate for BOTH the self-healing system and the baseline. This is the core comparison for your PS.

### Experiment 2: HotpotQA by question type
Edit `run_evaluation.py` or call programmatically:
```python
from data.hotpotqa_loader import load_hotpotqa
bridge_samples = [s for s in load_hotpotqa(500) if s["type"] == "bridge"][:100]
comparison_samples = [s for s in load_hotpotqa(500) if s["type"] == "comparison"][:100]
```

### Experiment 3: RAGBench — Hallucination Detection
```bash
python run_evaluation.py --dataset ragbench --subset covidqa --n_samples 200
python run_evaluation.py --dataset ragbench --subset hotpotqa --n_samples 200
python run_evaluation.py --dataset ragbench --subset msmarco --n_samples 200
```
**This gives you:** hallucination_detection_accuracy — how well the self-monitor identifies bad responses.

### Experiment 4: All datasets together
```bash
python run_evaluation.py --dataset both --n_samples 100 --compare_baseline
```

---

## Phase 4 — Collect Results (Day 3)

All results are saved to `results/` as JSON files:
```
results/
  hotpotqa_results.json      ← per-sample results + aggregate metrics
  ragbench_covidqa_results.json
  ragbench_hotpotqa_results.json
  summary.json               ← cross-dataset summary
```

### Key numbers to report in your paper/presentation:

| Metric | Where it comes from |
|--------|---------------------|
| EM (Exact Match) | hotpotqa_results.json → metrics.em |
| F1 | hotpotqa_results.json → metrics.f1 |
| Hallucination Rate | Any results → metrics.hallucination_rate |
| Correction Rate | Any results → metrics.correction_rate |
| Hallucination Detection Accuracy | ragbench_*_results.json → metrics.hallucination_detection_accuracy |
| Baseline EM | hotpotqa_results.json → metrics.baseline_em (with --compare_baseline) |
| EM Improvement | hotpotqa_results.json → metrics.em_improvement |

---

## Phase 5 — What to Write in Your Report

### Section 4 (Architecture) — already designed, use the diagram from the PS doc

### Section 5 (Implementation) — describe:
- `HybridRetriever`: BM25 (30%) + FAISS dense (70%) retrieval
- `TaskPlanner`: regex + LLM-based multi-hop decomposition
- `SelfMonitor`: 4-check failure detection (confidence, grounding, completeness, consistency)
- `CorrectiveActions`: 3 strategies (reformulate, re-retrieve, decompose)

### Section 6 (Results) — fill in from your JSON files:

**Table 1: Answer Quality (HotpotQA)**
| System | EM | F1 | Halluc. Rate |
|--------|----|----|-------------|
| Baseline RAG | baseline_em | baseline_f1 | ~0.3-0.5 |
| Self-Healing RAG | em | f1 | hallucination_rate |

**Table 2: RAGBench Faithfulness**
| Subset | Halluc. Detection Acc. | System Grounded Rate |
|--------|----------------------|---------------------|
| covidqa | ... | ... |
| hotpotqa | ... | ... |

**Table 3: Self-Healing Effectiveness**
| Metric | Value |
|--------|-------|
| Failure detection rate | failure_rate |
| Correction success rate | correction_rate |
| Avg correction rounds | (from per-sample data) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `HuggingFace model download slow` | Run once, it caches locally |
| `OpenAI RateLimitError` | Add `time.sleep(1)` in evaluator loop, or use `--n_samples 50` |
| `datasets not found` | `pip install datasets` |
| `No module named 'rank_bm25'` | `pip install rank-bm25` |
| `CUDA out of memory` | Set `LLM_BACKEND=openai` or use smaller HF model |

---

## File Map — What Each File Does

```
pipeline.py              ← START HERE — the main orchestrator
agent/llm_reasoner.py    ← all LLM calls go through here
agent/self_monitor.py    ← THE KEY FILE — failure detection logic
agent/corrective_actions.py  ← THE KEY FILE — self-healing logic
agent/task_planner.py    ← multi-hop question decomposition
retrieval/retriever.py   ← BM25 + FAISS hybrid retrieval
evaluation/metrics.py    ← EM, F1, faithfulness metrics
evaluation/evaluator.py  ← runs evaluation on datasets
data/hotpotqa_loader.py  ← loads HotpotQA
data/ragbench_loader.py  ← loads RAGBench
run_evaluation.py        ← ENTRY POINT — run this
demo.py                  ← quick test, no API key needed
config.py                ← all settings in one place
```
