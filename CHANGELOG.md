# Changelog

## 2.0.0 — 2026-10

A reliability and evaluation overhaul. The architecture (planner → hybrid retrieval → tiered
monitor → adaptive correction) is unchanged; the implementation now runs, is tested, and its
evaluation measures what it claims to.

### Fixed — blocking bugs
- `pipeline.py` read the planner's `PlanResult` dataclass as a dict (`plan['strategy']`) → `TypeError` on every query.
- `demo.py`, `ui.py` and the evaluator read `PipelineResult` as a dict (`output["final_answer"]`, `.get(...)`) → crash.
- The corrector read `RetrievalResult` objects as dicts (`d["text"]`) → crash on every correction.
- Baselines imported a non-existent module (`pipeline.pipeline`) and called missing methods.
- Experiment runner unpacked a 3-tuple into 2 values; its TaskPlanner patch had the wrong signature.
- Both default models were shut down by their providers (`gemini-2.0-flash-lite` on 2026-06-01,
  Groq `llama-3.1-8b-instant` on 2026-08-16).
- `faiss-cpu` was missing from `requirements.txt`; `numpy==1.26.5` does not exist; several pins conflicted.

### Fixed — correctness
- The final correction was never verified; now every candidate is checked and the best-grounded one is returned.
- Failure type was never passed to the corrector, so the failure→strategy map was dead code.
- Tier 2 NLI: answers of ≤ 2 words skipped the check (almost all HotpotQA answers); premise and hypothesis were glued
  into one string with a RoBERTa separator; passages were cut to 2,000 characters total.
- Tier 3 judge saw only 600 characters of context.
- The generator's context was cut at a fixed length, silently dropping the last retrieved passages.
- The "consistency" check flagged legitimate yes/no comparisons (`yes` + `no` sub-answers) as contradictions.
- Bridge decomposition answered hop 2 without the hop-1 entity.
- Config overrides never reached modules that used `from config import X`, so ablations were no-ops.
- All seeds sampled the same questions.
- RAGBench "hallucination detection accuracy" compared the monitor's flag on the system's own answer with the label of
  a different answer; it is now a proper detector benchmark on the labelled responses.
- Plots substituted `np.random.random()` for missing metrics; replaced with real tables/plots.
- API errors were returned as the answer "I don't know.", which looked like a model refusal in the metrics.
- EM/F1 now follow the official HotpotQA normalisation; supporting-fact matching uses exact titles.
- `scipy.stats.binom_test` (removed in SciPy 1.12) replaced by `binomtest`.

### Added
- One OpenAI-compatible client for Gemini / Groq / OpenAI / Ollama; retries with back-off and `Retry-After`; token
  accounting; reasoning-model support (`reasoning_effort`, output-token floor).
- Monitor: answer-type check, multi-hop consistency check, sentence-level NLI for long answers, joint premises,
  configurable tiers (`MonitorSettings` modes).
- Corrector: targeted re-retrieval that excludes already-seen passages, `#1` substitution, `adopt_last_hop`, futility
  checks, no repeated strategies.
- Stateless, thread-safe pipeline with dependency injection; per-query cost tracking via `ContextVar`.
- `verified` flag on every result (did the returned answer pass the monitor?) and accuracy on verified vs
  unverified answers in the reports (selective answering).
- FastAPI service (`/v1/ask`, `/v1/verify`, `/health`), Dockerfile, docker-compose.
- Evaluations: paired McNemar / bootstrap tests, healing analysis (fixed vs broken answers), resumable JSONL
  checkpoints, detector benchmark, retrieval benchmark, error analysis and case studies, README results updater.
- Baselines share models through a registry: vanilla, planner-only, NLI-only, LLM-judge-only, always-correct, CRAG.
- 98 offline tests (fakes for the LLM, embedder and NLI model), GitHub Actions CI on Python 3.10–3.12, ruff.
- Colab notebook to reproduce every result.
- Document chunking for long inputs; closed-book baseline.

### Removed
- `data/nq_loader.py`, `data/popqa_loader.py` (no passages to retrieve from, unused).
- Stub analysis modules (`result_analyzer.py`, `case_studies.py`, `visualizations.py`, `cost_analysis.py`).
- Unused modules: `paper/` (LaTeX pseudocode), `evaluation/faithfulness.py`, `utils/helpers.py`, and the
  per-baseline wrapper files (`vanilla_rag.py`, `always_correct.py`, `llm_judge_only.py`) now covered by the
  registry in `baselines/__init__.py`.
- `WHAT_TO_DO.md`; `SETUP_*.md` merged into `docs/BACKENDS.md`; `REPORT.md` moved to `docs/`.
- `google-generativeai`, `groq`, `requests`, `pandas`, `seaborn`, `scikit-learn`, `nltk` dependencies.

### Changed
- Default `TOP_K_RETRIEVAL` 7 → 5: start with a smaller, cheaper context and let healing add passages when the
  monitor finds a problem. Run `python -m evaluation.benchmark_retrieval` to see recall@k for your data.
