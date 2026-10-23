# Design and Evaluation of a Self-Healing Agentic RAG System

*Technical report — v2.0*

## 1. Problem

Retrieval-augmented generation (RAG) grounds an LLM in retrieved documents, but a RAG system
still answers confidently when retrieval missed the evidence, when a multi-hop chain breaks, or
when the model ignores the context. Standard RAG has no way to notice these failures.

This project builds a RAG system that **checks its own answer before returning it** and, when the
check fails, **chooses a repair strategy based on the kind of failure**. The design goal is to make
the checking cheap: most answers should be verified without an extra LLM call.

## 2. Related work

| Work | Idea | Relation to this project |
|---|---|---|
| Lewis et al., 2020 — RAG | retrieve-then-generate | the base pattern |
| Yang et al., 2018 — HotpotQA | multi-hop QA with supporting facts | main QA benchmark |
| Trivedi et al., 2022 — MuSiQue | 2–4 hop composed questions | harder QA benchmark (loader included) |
| Press et al., 2023 — Self-Ask | explicit follow-up sub-questions | `#1` substitution for bridge questions |
| Yao et al., 2023 — ReAct; Shinn et al., 2023 — Reflexion | act / reflect loops | the monitor → correct loop |
| Asai et al., 2024 — Self-RAG | learned reflection tokens | we use off-the-shelf models instead of fine-tuning |
| Yan et al., 2024 — CRAG | grade retrieved documents, re-retrieve | implemented as a baseline (`baselines/crag_baseline.py`) |
| Manakul et al., 2023 — SelfCheckGPT | detect hallucinations without references | motivation for a hallucination detector |
| Friel et al., 2024 — RAGBench | RAG responses labelled for groundedness | detector benchmark |

## 3. Architecture

```
question + documents
  → Planner           direct | sequential (bridge) | parallel (comparison); LLM decomposition
  → Hybrid retriever   BM25 + all-MiniLM-L6-v2, min-max linear fusion or RRF
  → Generator          chain-of-thought prompt, short-answer extraction
  → Monitor            Tier 1 heuristics → Tier 2 NLI → Tier 3 LLM judge (only if uncertain)
  → Corrector          strategy chosen by failure type, never repeated, skipped if futile
  → loop (≤ 2 corrections) → return the best-grounded candidate
```

Every candidate answer — including the one produced by the last correction — is checked by the
monitor. The returned answer is the best candidate by (passed the monitor, monitor confidence,
earliest round), so a correction that makes things worse cannot replace a better earlier answer.
If no candidate passes, the best one is returned with `verified = false`, so a caller can choose to
abstain instead of showing a possibly hallucinated answer.

### 3.1 Monitor tiers

| Tier | Check | Cost | Outcome |
|---|---|---|---|
| 1 | empty answer, refusal phrases, yes/no vs entity type mismatch, multi-hop inconsistency (final answer ≠ last hop's answer), long answer with no question keywords | ~0 ms | fail fast |
| 2 | NLI cross-encoder (`nli-deberta-v3-small`, ~70 M params): P(entailment) of the answer, phrased as a claim, against every passage window plus one joint premise of the top-2 passages. Long answers are checked sentence by sentence; the weakest sentence decides | ~50 ms CPU | pass if p > 0.7, fail if p < 0.2 |
| 3 | LLM judge with the full numbered context: `VERDICT: SUPPORTED/UNSUPPORTED` | 1 API call | only for 0.2 ≤ p ≤ 0.7 |

Thresholds are in `config.py` (`NLI_LOW`, `NLI_HIGH`). Each tier can be disabled, which yields the
ablations and baselines (`heuristic`, `nli`, `heuristic_nli`, `heuristic_llm`, `llm`, `tiered`).

### 3.2 Corrective strategies

| Failure type | First strategy | What it does |
|---|---|---|
| refusal | reformulate | LLM rephrases the question, retrieve with it, answer the original |
| ungrounded / irrelevant | re_retrieve | LLM writes a query for the *missing* evidence; passages already in context are excluded |
| hallucinated (judge) / empty | expand_retrieve | add passages about the question and the current answer (bridge entity) |
| type mismatch | decompose | sub-questions; for bridge questions hop 2 contains `#1`, replaced by hop 1's answer |
| inconsistent | adopt_last_hop | use the last hop's answer (no LLM call), then re-verify |

If the first strategy was already tried the next untried one is used
(re_retrieve → decompose → expand_retrieve → reformulate). Retrieval strategies are skipped when
every passage is already in context; if nothing useful remains, the loop stops early.

## 4. Evaluation methodology

* **QA (HotpotQA distractor, MuSiQue).** Official HotpotQA EM/F1 normalisation, 95 % bootstrap CIs,
  McNemar's test (EM) and a paired bootstrap test (F1) against vanilla RAG on the same questions.
  Cost is measured at the LLM client (calls, prompt/completion tokens) and latency per component.
* **Healing analysis.** Because the initial answer is recorded, we report how often the monitor
  flags an initial answer that is actually wrong (precision / recall), how many wrong answers
  healing fixed and how many right answers it broke, and EM on verified vs unverified answers
  (selective answering).
* **Hallucination detection (RAGBench).** The monitor judges RAGBench's *given* responses; labels
  come from `adherence_score`. We report precision, recall, F1, balanced accuracy, AUROC and LLM
  calls per response for each monitor configuration. (v1 compared the monitor's flag on the
  system's own answer with the label of a different answer, which does not measure detection.)
* **Retrieval.** BM25 vs dense vs linear fusion vs RRF on HotpotQA: recall@k, all-gold@k, MRR.
* **Ablations.** 15 YAML configs in `experiments/configs/` (no healing, no NLI, no judge, judge on
  everything, always-correct, no decomposition, rounds 1/3, top-k 3/7, BM25/dense/RRF, closed-book).

## 5. Results

Results are produced by the scripts above and pasted into the README by
`analysis/update_readme.py`; see `notebooks/reproduce_results.ipynb` for a one-click Colab run.
No numbers are reported here that were not produced by those scripts.

## 6. Limitations

* The claim template used for short answers ("The answer to the question … is X.") is a heuristic;
  NLI models handle yes/no and numeric comparisons poorly, so those usually escalate to the judge.
* The LLM judge is the same model as the generator, so it may share its blind spots
  (self-preference bias). A different judge model is a one-line config change worth testing.
* Thresholds were set by hand; they could be calibrated on the RAGBench training split.
* The benchmarks use a small per-question corpus (HotpotQA distractor). An open-corpus setting
  (fullwiki) would need a persistent vector database.

## References

1. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP.
3. Trivedi et al. (2022). MuSiQue: Multihop Questions via Single-hop Question Composition. TACL.
4. Press et al. (2023). Measuring and Narrowing the Compositionality Gap in Language Models. Findings of EMNLP.
5. Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR.
6. Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS.
7. Asai et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR.
8. Yan et al. (2024). Corrective Retrieval Augmented Generation. arXiv:2401.15884.
9. Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection. EMNLP.
10. Friel, Belyi & Sanyal (2024). RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems. arXiv:2407.11005.
11. Cormack, Clarke & Büttcher (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
12. Reimers & Gurevych (2019). Sentence-BERT. EMNLP.
