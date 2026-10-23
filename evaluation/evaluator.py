"""
evaluation/evaluator.py — Runs systems over a dataset and scores them.

Two evaluation modes
--------------------
1. QA (HotpotQA / MuSiQue / RAGBench-QA): every system answers every
   question; we report EM/F1 with bootstrap CIs, paired significance tests
   against the reference system, cost, and a *healing analysis* (how often the
   monitor flagged a wrong answer, how many wrong answers healing fixed and
   how many right answers it broke).

2. Detector (RAGBench): the self-monitor is run on RAGBench's *given*
   responses and compared with the human/GPT-4 grounding labels. This is the
   correct way to measure hallucination detection (the original code compared
   the monitor's flag on its own answer with the label of a different answer).

Progress is checkpointed to JSONL after every sample, so long runs on
rate-limited free tiers can be stopped and resumed.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import config
from agent.cost_tracker import CURRENT_TRACKER, CostTracker, aggregate_costs
from evaluation.metrics import (
    best_over_golds,
    binary_classification_metrics,
    contains_match,
    exact_match,
    retrieval_recall_at_k,
    roc_auc,
    rouge_l,
    supporting_doc_f1,
    token_f1,
)
from evaluation.statistical import bootstrap_confidence_interval, mcnemar_test, paired_bootstrap_test
from retrieval.retriever import extract_title

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore
        return it


# ── Helpers ──────────────────────────────────────────────────────────────────
def sample_documents(sample: Dict[str, Any]) -> List[str]:
    """Documents for a sample as 'Title: ...\\n<text>' strings."""
    if "documents" in sample:
        return list(sample["documents"])
    return [f"Title: {d['title']}\n{d['text']}" for d in sample.get("context_docs", [])]


def sample_golds(sample: Dict[str, Any]) -> List[str]:
    gold = sample.get("answer") or sample.get("gold_response") or ""
    return [gold, *sample.get("answer_aliases", [])]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ── Scoring a single QA prediction ───────────────────────────────────────────
def score_qa_record(sample: Dict[str, Any], output: Any, system: str, long_answers: bool,
                    latency: float) -> Dict[str, Any]:
    golds = sample_golds(sample)
    pred, init = output.final_answer, output.initial_answer
    titles = [extract_title(d) for d in output.retrieved_docs]
    gold_titles = sample.get("supporting_titles", [])
    rec = {
        "id": sample.get("id", ""),
        "system": system,
        "question": sample["question"],
        "question_type": sample.get("type", "unknown") or "unknown",
        "gold": golds[0],
        "answer": pred,
        "initial_answer": init,
        "em": best_over_golds(exact_match, pred, golds),
        "f1": best_over_golds(token_f1, pred, golds),
        "contains": best_over_golds(contains_match, pred, golds),
        "initial_em": best_over_golds(exact_match, init, golds),
        "initial_f1": best_over_golds(token_f1, init, golds),
        "failure_detected": bool(output.failure_detected),
        "failure_type": output.failure_type,
        "correction_attempted": bool(output.correction_attempted),
        "n_correction_rounds": output.n_correction_rounds,
        "selected_round": output.selected_round,
        "verified": output.verified,
        "initial_monitor_tier": output.initial_monitor_tier,
        "strategy": output.strategy,
        "rounds": [r if isinstance(r, dict) else r.__dict__ for r in output.rounds],
        "cost": output.cost_data,
        "latency_s": latency,
        "error": None,
    }
    if gold_titles:
        rec["sup_f1"] = supporting_doc_f1(titles, gold_titles)
        rec["support_recall"] = retrieval_recall_at_k(titles, gold_titles)
    if long_answers:
        rec["rouge_l"] = rouge_l(pred, golds[0])
    return rec


def _is_correct(rec: Dict[str, Any], key: str = "em", long_answers: bool = False) -> bool:
    if long_answers:
        return rec.get("f1" if key == "em" else "initial_f1", 0.0) >= 0.5
    return rec.get(key, 0.0) >= 1.0


# ── Aggregation ──────────────────────────────────────────────────────────────
def summarize_qa(records: List[Dict[str, Any]], long_answers: bool = False) -> Dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    n = len(ok)
    out: Dict[str, Any] = {"n_samples": n, "n_errors": len(records) - n}
    if n == 0:
        return out
    for key in ("em", "f1", "contains", "sup_f1", "support_recall", "rouge_l"):
        vals = [r[key] for r in ok if key in r]
        if vals:
            mean, lo, hi = bootstrap_confidence_interval(vals)
            out[key] = mean
            out[f"{key}_ci"] = [lo, hi]
    out["avg_latency_s"] = _mean(r["latency_s"] for r in ok)
    out.update({k: v for k, v in aggregate_costs([r["cost"] for r in ok]).items() if k != "n_queries"})
    out["failure_detected_rate"] = _mean(r["failure_detected"] for r in ok)
    verified = [r for r in ok if r.get("verified") is not None]
    if verified:
        # Selective answering: accuracy on the answers the monitor vouched for.
        out["verified_rate"] = _mean(r["verified"] for r in verified)
        vouched = [r for r in verified if r["verified"]]
        out["em_when_verified"] = _mean(r["em"] for r in vouched) if vouched else None
        unvouched = [r for r in verified if not r["verified"]]
        out["em_when_unverified"] = _mean(r["em"] for r in unvouched) if unvouched else None
    out["correction_attempted_rate"] = _mean(r["correction_attempted"] for r in ok)

    # Healing analysis: did the monitor flag the right answers, and did healing help?
    monitored = [r for r in ok if any(rd.get("has_failure") is not None for rd in r.get("rounds", []))]
    if monitored:
        init_wrong = [r for r in monitored if not _is_correct(r, "initial_em", long_answers)]
        init_right = [r for r in monitored if _is_correct(r, "initial_em", long_answers)]
        flagged = [r for r in monitored if r["failure_detected"]]
        fixed = [r for r in init_wrong if r["failure_detected"] and _is_correct(r, "em", long_answers)]
        broken = [r for r in init_right if r["failure_detected"] and not _is_correct(r, "em", long_answers)]
        flagged_wrong = [r for r in flagged if not _is_correct(r, "initial_em", long_answers)]
        out["healing"] = {
            "n_monitored": len(monitored),
            "initial_accuracy": len(init_right) / len(monitored),
            # None (rendered as "–") when undefined, instead of a misleading 0.0
            "monitor_precision": len(flagged_wrong) / len(flagged) if flagged else None,
            "monitor_recall": (sum(1 for r in init_wrong if r["failure_detected"]) / len(init_wrong)
                               if init_wrong else None),
            "fixed": len(fixed),
            "broken": len(broken),
            "fix_rate": len(fixed) / max(1, sum(1 for r in init_wrong if r["failure_detected"])),
            "break_rate": len(broken) / max(1, sum(1 for r in init_right if r["failure_detected"])),
            "net_gain": (len(fixed) - len(broken)) / len(monitored),
        }

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok:
        by_type.setdefault(r.get("question_type", "unknown"), []).append(r)
    if len(by_type) > 1:
        out["per_question_type"] = {
            t: {"n": len(rs), "em": _mean(r["em"] for r in rs), "f1": _mean(r["f1"] for r in rs)}
            for t, rs in sorted(by_type.items())
        }
    return out


def compare_systems(records: Dict[str, List[Dict[str, Any]]], reference: str) -> Dict[str, Any]:
    """Paired tests of every system against the reference on the questions both answered."""
    ref = {r["id"]: r for r in records.get(reference, []) if not r.get("error")}
    out = {}
    for name, recs in records.items():
        if name == reference:
            continue
        mine = {r["id"]: r for r in recs if not r.get("error")}
        ids = sorted(set(ref) & set(mine))
        if not ids:
            continue
        em_a = [mine[i]["em"] for i in ids]
        em_b = [ref[i]["em"] for i in ids]
        f1_test = paired_bootstrap_test([mine[i]["f1"] for i in ids], [ref[i]["f1"] for i in ids])
        mc = mcnemar_test([e >= 1 for e in em_a], [e >= 1 for e in em_b])
        out[name] = {
            "n_paired": len(ids),
            "delta_em": _mean(em_a) - _mean(em_b),
            "delta_f1": f1_test["mean_diff"],
            "em_mcnemar_p": mc["p_value"],
            "f1_bootstrap_p": f1_test["p_value"],
            "wins": mc["a_only"],
            "losses": mc["b_only"],
            "delta_llm_calls": _mean(mine[i]["cost"].get("n_llm_calls", 0) for i in ids)
            - _mean(ref[i]["cost"].get("n_llm_calls", 0) for i in ids),
        }
    return out


def summarize_detector(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    if not ok:
        return {"n_samples": 0, "n_errors": len(records)}
    y_true = [r["label_hallucinated"] for r in ok]
    y_pred = [r["flagged"] for r in ok]
    out = {"n_samples": len(ok), "n_errors": len(records) - len(ok),
           "positive_rate": _mean(y_true), "flag_rate": _mean(y_pred)}
    out.update(binary_classification_metrics(y_true, y_pred))
    out["auroc"] = roc_auc(y_true, [r["score"] for r in ok])
    out["avg_llm_calls"] = _mean(r["llm_calls"] for r in ok)
    out["avg_latency_s"] = _mean(r["latency_s"] for r in ok)
    tiers = [r["tier"] for r in ok]
    out["tier_distribution"] = {f"tier_{t}": tiers.count(t) / len(tiers) for t in range(4)}
    return out


# ── Evaluator ────────────────────────────────────────────────────────────────
class Evaluator:
    def __init__(self, results_dir: Optional[str] = None) -> None:
        self.results_dir = results_dir or config.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

    def _run_dir(self, tag: str) -> str:
        path = os.path.join(self.results_dir, tag)
        os.makedirs(path, exist_ok=True)
        return path

    def evaluate_qa(
        self,
        systems: Dict[str, Any],
        samples: Sequence[Dict[str, Any]],
        tag: str,
        long_answers: bool = False,
        resume: bool = True,
        reference: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        summary_name: str = "summary",
    ) -> Dict[str, Any]:
        """Run every system on every sample; returns (and saves) the summary."""
        run_dir = self._run_dir(tag)
        records: Dict[str, List[Dict[str, Any]]] = {}
        for name, system in systems.items():
            path = os.path.join(run_dir, f"{name}.jsonl")
            if not resume and os.path.exists(path):
                os.remove(path)
            done = {r["id"]: r for r in _read_jsonl(path) if not r.get("error")}
            # Drop failed rows so they are retried on resume.
            if resume and os.path.exists(path):
                with open(path, "w", encoding="utf-8") as fh:
                    for r in done.values():
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            todo = [s for s in samples if s.get("id", "") not in done]
            if done:
                logger.info("[%s] resuming: %d done, %d to go", name, len(done), len(todo))
            for sample in tqdm(todo, desc=name):
                start = time.perf_counter()
                try:
                    out = system.run(sample["question"], sample_documents(sample),
                                     question_type=sample.get("type") if sample.get("type") in
                                     ("bridge", "comparison") else None)
                    rec = score_qa_record(sample, out, name, long_answers, time.perf_counter() - start)
                except KeyboardInterrupt:
                    raise
                except Exception as err:  # keep going; failed rows are retried on resume
                    logger.error("[%s] sample %s failed: %s", name, sample.get("id"), err)
                    rec = {"id": sample.get("id", ""), "system": name, "error": f"{type(err).__name__}: {err}",
                           "latency_s": time.perf_counter() - start}
                _append_jsonl(path, rec)
                done[rec["id"]] = rec
            ids = {s.get("id", "") for s in samples}
            records[name] = [r for r in done.values() if r["id"] in ids]

        reference = reference or ("vanilla_rag" if "vanilla_rag" in systems else next(iter(systems)))
        summary = {
            "tag": tag,
            "mode": "qa",
            "long_answers": long_answers,
            "reference": reference,
            "meta": meta or {},
            "systems": {name: summarize_qa(recs, long_answers) for name, recs in records.items()},
            "comparisons": compare_systems(records, reference),
        }
        self._save_summary(run_dir, summary, summary_name)
        return summary

    def evaluate_detector(
        self,
        monitors: Dict[str, Any],
        samples: Sequence[Dict[str, Any]],
        tag: str,
        resume: bool = True,
        meta: Optional[Dict[str, Any]] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Score RAGBench responses with each monitor configuration against the gold labels."""
        run_dir = self._run_dir(tag)
        results = {}
        for mode, monitor in monitors.items():
            path = os.path.join(run_dir, f"detector_{mode}.jsonl")
            if not resume and os.path.exists(path):
                os.remove(path)
            done = {r["id"]: r for r in _read_jsonl(path) if not r.get("error")}
            for sample in tqdm([s for s in samples if s["id"] not in done], desc=f"detector/{mode}"):
                tracker = CostTracker()
                token = CURRENT_TRACKER.set(tracker)
                start = time.perf_counter()
                try:
                    report = monitor.evaluate(sample["question"], sample["gold_response"], sample["documents"])
                    rec = {
                        "id": sample["id"], "mode": mode,
                        "label_hallucinated": not sample["adherence_score"],
                        "flagged": report.has_failure,
                        "score": 1.0 - float(report.confidence_score),  # higher = more likely hallucinated
                        "tier": report.monitor_tier_used,
                        "failure_type": report.failure_type,
                        "nli_entailment": report.nli_entailment,
                        "judge_verdict": report.judge_verdict,
                        "llm_calls": tracker.current.n_llm_calls,
                        "latency_s": time.perf_counter() - start,
                        "error": None,
                    }
                except KeyboardInterrupt:
                    raise
                except Exception as err:
                    logger.error("[detector/%s] sample %s failed: %s", mode, sample.get("id"), err)
                    rec = {"id": sample["id"], "mode": mode, "error": f"{type(err).__name__}: {err}"}
                finally:
                    CURRENT_TRACKER.reset(token)
                _append_jsonl(path, rec)
                done[rec["id"]] = rec
            ids = {s["id"] for s in samples}
            results[mode] = summarize_detector([r for r in done.values() if r["id"] in ids])
        summary = {"tag": tag, "mode": "detector", "meta": meta or {}, "monitors": results}
        self._save_summary(run_dir, summary)
        return summary

    def _save_summary(self, run_dir: str, summary: Dict[str, Any], name: str = "summary") -> None:
        with open(os.path.join(run_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        try:
            from analysis.report import render_markdown

            report = "report.md" if name == "summary" else f"report_{name}.md"
            with open(os.path.join(run_dir, report), "w", encoding="utf-8") as fh:
                fh.write(render_markdown(summary))
        except Exception as err:  # reporting must never lose results
            logger.warning("Could not render report.md: %s", err)
        logger.info("Saved results to %s", run_dir)
