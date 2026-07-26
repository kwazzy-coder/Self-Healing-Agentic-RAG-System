"""
evaluation/evaluator.py — Clean evaluation with logging and statistical rigor.
"""
import json
import os
import time
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from pipeline import SelfHealingRAGPipeline
from evaluation.metrics import (
    exact_match, token_f1, rouge_l, bert_score_metric, contains_match,
    supporting_doc_f1, retrieval_recall_at_k, retrieval_mrr,
    compute_aggregate_metrics, compute_per_type_metrics, format_results_table,
    print_metrics
)
from evaluation.statistical import bootstrap_confidence_interval, format_with_ci
from config import RESULTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for assessing RAG pipeline performance.
    """
    def __init__(self, pipeline: SelfHealingRAGPipeline):
        self.pipeline = pipeline
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def evaluate_hotpotqa(self, samples: List[Dict[str, Any]], compare_baseline: bool = False) -> Dict[str, Any]:
        """
        Evaluate the pipeline on HotpotQA dataset.
        """
        logger.info(f"Running on {len(samples)} HotpotQA samples...")
        results = []

        for sample in tqdm(samples, desc="HotpotQA"):
            start_time = time.time()
            
            docs = [f"Title: {d['title']}\n{d['text']}" for d in sample["context_docs"]]
            gold_answer = sample["answer"]
            gold_support = list(set(sample["supporting_titles"]))
            q_type = sample.get("type", "unknown")

            # Baseline
            base_em, base_f1, base_rouge, base_answer = 0.0, 0.0, 0.0, ""
            base_failure = False
            if compare_baseline:
                base = self.pipeline.run_baseline(question=sample["question"], documents=docs)
                base_answer = base.get("answer", "")
                base_em = exact_match(base_answer, gold_answer)
                base_f1 = token_f1(base_answer, gold_answer)
                base_rouge = rouge_l(base_answer, gold_answer)
                # For baseline, failure is generally assumed false unless reported
                base_failure = base.get("failure_detected", False) 

            # Self-Healing
            output = self.pipeline.run(
                question=sample["question"],
                documents=docs,
                question_type=q_type,
            )
            
            latency = time.time() - start_time
            pred_answer = output.get("final_answer", "")
            
            retrieved_titles = self._extract_titles(output.get("retrieved_docs", []))
            
            em = exact_match(pred_answer, gold_answer)
            f1 = token_f1(pred_answer, gold_answer)
            rouge = rouge_l(pred_answer, gold_answer)
            contains = contains_match(pred_answer, gold_answer)
            bert = bert_score_metric(pred_answer, gold_answer)
            
            sup_f1 = supporting_doc_f1(retrieved_titles, gold_support)
            rec_at_k = retrieval_recall_at_k(retrieved_titles, gold_support, k=3)
            mrr = retrieval_mrr(retrieved_titles, gold_support)

            correction_succeeded = (
                output.get("correction_attempted", False) and em >= base_em
            ) if compare_baseline else output.get("correction_attempted", False)

            res_dict = {
                "id": sample["id"],
                "question": sample["question"],
                "question_type": q_type,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "baseline_answer": base_answer,
                "em": em, 
                "f1": f1,
                "rouge_l": rouge,
                "contains": contains,
                "bert_score_f1": bert.get("f1", 0.0),
                "sup_f1": sup_f1,
                "recall_at_k": rec_at_k,
                "mrr": mrr,
                "baseline_em": base_em,
                "baseline_f1": base_f1,
                "baseline_rouge": base_rouge,
                "baseline_failure": base_failure,
                "has_failure": output.get("failure_detected", False),
                "correction_attempted": output.get("correction_attempted", False),
                "correction_succeeded": correction_succeeded,
                "n_correction_rounds": output.get("n_correction_rounds", 0),
                "failure_reason": output.get("failure_reason", ""),
                "self_healing_helped": em > base_em if compare_baseline else False,
                "self_healing_hurt": em < base_em if compare_baseline else False,
                "latency_s": latency
            }
            results.append(res_dict)

        # Overall Metrics
        metrics = compute_aggregate_metrics(results)
        
        # Per-Type Metrics
        type_metrics = compute_per_type_metrics(results)
        metrics["per_question_type"] = type_metrics
        
        # Add Confidence Intervals for key metrics
        for key in ["em", "f1", "rouge_l"]:
            scores = [r[key] for r in results]
            mean, lower, upper = bootstrap_confidence_interval(scores)
            metrics[f"{key}_ci"] = format_with_ci(mean, lower, upper)

        if compare_baseline:
            n = len(results)
            base_em_avg = sum(r["baseline_em"] for r in results) / n
            base_f1_avg = sum(r["baseline_f1"] for r in results) / n
            base_rouge_avg = sum(r["baseline_rouge"] for r in results) / n
            base_fail_avg = sum(r["baseline_failure"] for r in results) / n
            helped = [r for r in results if r.get("self_healing_helped")]
            hurt = [r for r in results if r.get("self_healing_hurt")]
            
            metrics.update({
                "baseline_em": base_em_avg,
                "baseline_f1": base_f1_avg,
                "baseline_rouge_l": base_rouge_avg,
                "baseline_failure_rate": base_fail_avg,
                "em_improvement": metrics.get("em", 0) - base_em_avg,
                "f1_improvement": metrics.get("f1", 0) - base_f1_avg,
                "net_improvement_rate": len(helped) / n,
                "net_hurt_rate": len(hurt) / n,
            })

        self._print_clean_metrics(metrics)
        
        if type_metrics:
            logger.info("  ── Per Question Type Breakdown ──")
            for qt, qtm in type_metrics.items():
                logger.info(f"  [{qt.upper()}] EM: {qtm.get('em', 0):.3f} | F1: {qtm.get('f1', 0):.3f} | Failure: {qtm.get('failure_detected_rate', 0):.3f}")

        if compare_baseline:
            self._print_correction_analysis(results)
            self._print_comparison_table(metrics, compare_baseline)

        self._save_results(results, metrics, "hotpotqa")
        return metrics

    def _print_clean_metrics(self, metrics: Dict[str, Any]):
        """Print metrics table exactly once, cleanly using logger."""
        logger.info(f"\n{'='*50}\n  HotpotQA Results\n{'='*50}")
        
        keys_order = [
            "n_samples", "em", "em_ci", "f1", "f1_ci", "rouge_l", "rouge_l_ci", 
            "bert_score_f1", "contains_match",
            "supporting_facts_f1", "retrieval_recall_at_k", "retrieval_mrr",
            "failure_detected_rate", "correction_attempted_rate",
            "correction_success_rate", "avg_latency_s",
            "baseline_em", "baseline_f1",
            "em_improvement", "f1_improvement",
            "net_improvement_rate", "net_hurt_rate",
        ]
        for k in keys_order:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    logger.info(f"  {k:<35}: {v:.4f}")
                else:
                    logger.info(f"  {k:<35}: {v}")
        logger.info(f"{'='*50}\n")

    def _print_correction_analysis(self, results: List[Dict[str, Any]]):
        """Print correction analysis."""
        n = len(results)
        if n == 0:
            return
            
        helped = sum(1 for r in results if r.get("self_healing_helped"))
        hurt = sum(1 for r in results if r.get("self_healing_hurt"))
        same = n - helped - hurt
        attempted = sum(1 for r in results if r["correction_attempted"])

        logger.info("  ── Correction Impact Analysis ──")
        logger.info(f"  Corrections attempted : {attempted}/{n}")
        logger.info(f"  Self-healing HELPED   : {helped} samples")
        logger.info(f"  Self-healing HURT     : {hurt} samples")
        logger.info(f"  No change             : {same} samples")
        logger.info(f"  Net beneficial rate   : {helped/n:.1%}\n")

    def _print_comparison_table(self, metrics: Dict[str, Any], compare_baseline: bool):
        """Print clean comparison table using standard markdown layout."""
        sh_em = metrics.get("em", 0)
        sh_f1 = metrics.get("f1", 0)
        sh_rouge = metrics.get("rouge_l", 0)
        sh_cor = metrics.get("correction_success_rate", 0)

        metrics_list = [
            ("Self-Healing RAG", {
                "em": sh_em, "f1": sh_f1, "rouge_l": sh_rouge, "correction_success_rate": sh_cor
            })
        ]
        
        if compare_baseline:
            metrics_list.append(
                ("Baseline RAG", {
                    "em": metrics.get("baseline_em", 0),
                    "f1": metrics.get("baseline_f1", 0),
                    "rouge_l": metrics.get("baseline_rouge_l", 0),
                    "correction_success_rate": 0.0 # Baseline has no self-correction
                })
            )
            
        table_str = format_results_table(metrics_list)
        logger.info(f"Comparison Table:\n{table_str}")

    def evaluate_ragbench(self, samples: List[Dict[str, Any]], subset: str = "ragbench") -> Dict[str, Any]:
        """
        Evaluate the pipeline on RAGBench dataset.
        """
        logger.info(f"Running {len(samples)} RAGBench '{subset}' samples...")
        results = []

        for sample in tqdm(samples, desc=f"RAGBench/{subset}"):
            start_time = time.time()
            output = self.pipeline.run(
                question=sample["question"], documents=sample["documents"]
            )
            latency = time.time() - start_time
            pred = output.get("final_answer", "")
            
            f1_gold = token_f1(pred, sample["gold_response"])
            rouge = rouge_l(pred, sample["gold_response"])
            bert = bert_score_metric(pred, sample["gold_response"])
            
            gold_hallucinated = not sample.get("adherence_score", True)
            system_flagged = output.get("failure_detected", False)
            detection_correct = (system_flagged == gold_hallucinated)

            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "predicted": pred,
                "f1_vs_gold": f1_gold,
                "rouge_l": rouge,
                "bert_score_f1": bert.get("f1", 0.0),
                "has_failure": system_flagged,
                "correction_attempted": output.get("correction_attempted", False),
                "gold_is_hallucinated": gold_hallucinated,
                "system_flagged": system_flagged,
                "detection_correct": detection_correct,
                "system_grounded": not system_flagged,
                "latency_s": latency
            })

        n = len(results)
        if n == 0:
            return {}
            
        failures = [r for r in results if r["has_failure"]]
        attempted = [r for r in results if r["correction_attempted"]]
        f1_scores = [r["f1_vs_gold"] for r in results]
        
        tp = sum(1 for r in results if r["system_flagged"] and r["gold_is_hallucinated"])
        fp = sum(1 for r in results if r["system_flagged"] and not r["gold_is_hallucinated"])
        fn = sum(1 for r in results if not r["system_flagged"] and r["gold_is_hallucinated"])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_detect = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        latencies = [r["latency_s"] for r in results]

        metrics = {
            "n_samples": n,
            "f1_vs_gold_response": sum(f1_scores) / n,
            "failure_detected_rate": len(failures) / n,
            "correction_attempted_rate": len(attempted) / len(failures) if failures else 0.0,
            "system_grounded_rate": sum(1 for r in results if r["system_grounded"]) / n,
            "halluc_detection_precision": precision,
            "halluc_detection_recall": recall,
            "halluc_detection_f1": f1_detect,
            "halluc_detection_accuracy": sum(1 for r in results if r["detection_correct"]) / n,
            "avg_latency_s": sum(latencies) / n if latencies else 0.0
        }

        print_metrics(metrics, title=f"RAGBench/{subset} Results")
        self._save_results(results, metrics, f"ragbench_{subset}")
        return metrics

    def _extract_titles(self, docs: List[str]) -> List[str]:
        """Extract titles from document strings formatted as 'Title: <title>\n<text>'"""
        titles = []
        for d in docs:
            if d.startswith("Title:"):
                titles.append(d.split("\n")[0].replace("Title:", "").strip())
        return titles

    def _save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any], tag: str):
        """Save raw results and aggregate metrics to JSON."""
        path = os.path.join(RESULTS_DIR, f"{tag}_results.json")
        with open(path, "w") as f:
            json.dump({"metrics": metrics, "samples": results}, f, indent=2)
        logger.info(f"Results saved to {path}")
