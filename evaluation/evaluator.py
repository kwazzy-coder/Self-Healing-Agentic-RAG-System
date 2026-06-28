"""
evaluation/evaluator.py — Clean evaluation with no duplicate printing.
"""
import json, os
from tqdm import tqdm
from pipeline import SelfHealingRAGPipeline
from evaluation.metrics import (
    exact_match, token_f1, supporting_doc_f1, print_metrics
)
from config import RESULTS_DIR


class Evaluator:
    def __init__(self, pipeline: SelfHealingRAGPipeline):
        self.pipeline = pipeline
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def evaluate_hotpotqa(self, samples, compare_baseline=False):
        print(f"\n[Evaluator] Running on {len(samples)} HotpotQA samples...")
        results = []

        for sample in tqdm(samples, desc="HotpotQA"):
            docs         = [f"Title: {d['title']}\n{d['text']}"
                            for d in sample["context_docs"]]
            gold_answer  = sample["answer"]
            gold_support = list(set(sample["supporting_titles"]))

            # Baseline
            base_em, base_f1, base_answer = 0.0, 0.0, ""
            if compare_baseline:
                base = self.pipeline.run_baseline(
                    question=sample["question"], documents=docs)
                base_answer = base["answer"]
                base_em     = exact_match(base_answer, gold_answer)
                base_f1     = token_f1(base_answer, gold_answer)

            # Self-Healing
            output = self.pipeline.run(
                question=sample["question"],
                documents=docs,
                question_type=sample.get("type"),
            )

            retrieved_titles = self._extract_titles(output["retrieved_docs"])
            em     = exact_match(output["final_answer"], gold_answer)
            f1     = token_f1(output["final_answer"], gold_answer)
            sup_f1 = supporting_doc_f1(retrieved_titles, gold_support)

            correction_succeeded = (
                output["correction_attempted"] and em >= base_em
            ) if compare_baseline else output["correction_attempted"]

            results.append({
                "id":                   sample["id"],
                "question":             sample["question"],
                "gold_answer":          gold_answer,
                "predicted_answer":     output["final_answer"],
                "baseline_answer":      base_answer,
                "em": em, "f1": f1, "sup_f1": sup_f1,
                "baseline_em":          base_em,
                "baseline_f1":          base_f1,
                "has_failure":          output["failure_detected"],
                "correction_attempted": output["correction_attempted"],
                "correction_succeeded": correction_succeeded,
                "n_correction_rounds":  output["n_correction_rounds"],
                "failure_reason":       output.get("failure_reason", ""),
                "self_healing_helped":  em > base_em if compare_baseline else False,
                "self_healing_hurt":    em < base_em if compare_baseline else False,
            })

        metrics = self._compute_metrics(results, compare_baseline)

        # Print metrics ONCE cleanly
        self._print_clean_metrics(metrics)

        # Print correction analysis ONCE
        if compare_baseline:
            self._print_correction_analysis(results)

        # Print comparison table
        self._print_comparison_table(metrics, compare_baseline)

        self._save_results(results, metrics, "hotpotqa")
        return metrics

    def _compute_metrics(self, results, compare_baseline):
        n         = len(results)
        em_scores = [r["em"]  for r in results]
        f1_scores = [r["f1"]  for r in results]
        sup_f1s   = [r["sup_f1"] for r in results]
        failures  = [r for r in results if r["has_failure"]]
        attempted = [r for r in results if r["correction_attempted"]]
        succeeded = [r for r in results if r["correction_succeeded"]]

        m = {
            "n_samples":                  n,
            "em":                         sum(em_scores) / n,
            "f1":                         sum(f1_scores) / n,
            "failure_rate":               len(failures) / n,
            "correction_attempted_rate":  len(attempted) / len(failures)
                                          if failures else 0.0,
            "correction_success_rate":    len(succeeded) / len(attempted)
                                          if attempted else 0.0,
            "task_success_rate":          sum(em_scores) / n,
            "hallucination_rate":         len(failures) / n,
            "supporting_facts_f1":        sum(sup_f1s) / len(sup_f1s)
                                          if sup_f1s else 0.0,
        }

        if compare_baseline:
            base_em  = sum(r["baseline_em"] for r in results) / n
            base_f1  = sum(r["baseline_f1"] for r in results) / n
            helped   = [r for r in results if r.get("self_healing_helped")]
            hurt     = [r for r in results if r.get("self_healing_hurt")]
            m.update({
                "baseline_em":         base_em,
                "baseline_f1":         base_f1,
                "em_improvement":      m["em"] - base_em,
                "f1_improvement":      m["f1"] - base_f1,
                "net_improvement_rate": len(helped) / n,
                "net_hurt_rate":        len(hurt) / n,
            })
        return m

    def _print_clean_metrics(self, metrics):
        """Print metrics table exactly once, no duplicates."""
        print(f"\n{'='*50}")
        print(f"  HotpotQA Results")
        print(f"{'='*50}")
        # Which keys to show and in what order
        keys_order = [
            "n_samples", "em", "f1",
            "failure_rate", "correction_attempted_rate",
            "correction_success_rate", "task_success_rate",
            "hallucination_rate", "supporting_facts_f1",
            "baseline_em", "baseline_f1",
            "em_improvement", "f1_improvement",
            "net_improvement_rate", "net_hurt_rate",
        ]
        for k in keys_order:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    print(f"  {k:<35}: {v:.4f}")
                else:
                    print(f"  {k:<35}: {v}")
        print(f"{'='*50}\n")

    def _print_correction_analysis(self, results):
        """Print correction analysis exactly once."""
        n        = len(results)
        helped   = sum(1 for r in results if r.get("self_healing_helped"))
        hurt     = sum(1 for r in results if r.get("self_healing_hurt"))
        same     = n - helped - hurt
        attempted = sum(1 for r in results if r["correction_attempted"])

        print("  ── Correction Impact Analysis ──")
        print(f"  Corrections attempted : {attempted}/{n}")
        print(f"  Self-healing HELPED   : {helped} samples")
        print(f"  Self-healing HURT     : {hurt} samples")
        print(f"  No change             : {same} samples")
        print(f"  Net beneficial rate   : {helped/n:.1%}\n")

    def _print_comparison_table(self, metrics, compare_baseline):
        """Print clean comparison table."""
        sh_em  = metrics.get("em", 0)
        sh_f1  = metrics.get("f1", 0)
        sh_hal = metrics.get("hallucination_rate", 0)
        sh_cor = metrics.get("correction_success_rate", 0)  # use SUCCESS rate

        rows = [
            ["Self-Healing RAG", f"{sh_em:.3f}", f"{sh_f1:.3f}",
             f"{sh_hal:.3f}", f"{sh_cor:.3f}"],
        ]
        if compare_baseline:
            rows.append([
                "Baseline RAG",
                f"{metrics.get('baseline_em',0):.3f}",
                f"{metrics.get('baseline_f1',0):.3f}",
                "1.000", "0.000"
            ])

        headers    = ["System", "EM", "F1", "Halluc. Rate", "Corr. Success"]
        col_widths = [max(len(h), max(len(r[i]) for r in rows))
                      for i, h in enumerate(headers)]
        sep        = "+" + "+".join("-"*(w+2) for w in col_widths) + "+"
        hrow       = "| " + " | ".join(
            h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

        print(sep); print(hrow); print(sep)
        for row in rows:
            print("| " + " | ".join(
                v.ljust(w) for v, w in zip(row, col_widths)) + " |")
        print(sep)

    def evaluate_ragbench(self, samples, subset="ragbench"):
        print(f"\n[Evaluator] Running {len(samples)} RAGBench '{subset}' samples...")
        results = []

        for sample in tqdm(samples, desc=f"RAGBench/{subset}"):
            output  = self.pipeline.run(
                question=sample["question"], documents=sample["documents"])
            f1_gold = token_f1(output["final_answer"], sample["gold_response"])
            gold_hallucinated  = not sample.get("adherence_score", True)
            system_flagged     = output["failure_detected"]
            detection_correct  = (system_flagged == gold_hallucinated)

            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "predicted": output["final_answer"],
                "f1_vs_gold": f1_gold,
                "has_failure": output["failure_detected"],
                "correction_attempted": output["correction_attempted"],
                "gold_is_hallucinated": gold_hallucinated,
                "system_flagged": system_flagged,
                "detection_correct": detection_correct,
                "system_grounded": not output["failure_detected"],
            })

        n         = len(results)
        failures  = [r for r in results if r["has_failure"]]
        attempted = [r for r in results if r["correction_attempted"]]
        f1_scores = [r["f1_vs_gold"] for r in results]
        tp = sum(1 for r in results
                 if r["system_flagged"] and r["gold_is_hallucinated"])
        fp = sum(1 for r in results
                 if r["system_flagged"] and not r["gold_is_hallucinated"])
        fn = sum(1 for r in results
                 if not r["system_flagged"] and r["gold_is_hallucinated"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_detect = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        metrics = {
            "n_samples":                  n,
            "f1_vs_gold_response":        sum(f1_scores) / n,
            "failure_rate":               len(failures) / n,
            "correction_attempted_rate":  len(attempted) / len(failures)
                                          if failures else 0.0,
            "system_grounded_rate":       sum(1 for r in results
                                          if r["system_grounded"]) / n,
            "halluc_detection_precision": precision,
            "halluc_detection_recall":    recall,
            "halluc_detection_f1":        f1_detect,
            "halluc_detection_accuracy":  sum(1 for r in results
                                          if r["detection_correct"]) / n,
        }

        print_metrics(metrics, title=f"RAGBench/{subset} Results")
        self._save_results(results, metrics, f"ragbench_{subset}")
        return metrics

    def _extract_titles(self, docs):
        titles = []
        for d in docs:
            if d.startswith("Title:"):
                titles.append(d.split("\n")[0].replace("Title:","").strip())
        return titles

    def _save_results(self, results, metrics, tag):
        path = os.path.join(RESULTS_DIR, f"{tag}_results.json")
        with open(path, "w") as f:
            json.dump({"metrics": metrics, "samples": results}, f, indent=2)
        print(f"[Evaluator] Results saved to {path}")
