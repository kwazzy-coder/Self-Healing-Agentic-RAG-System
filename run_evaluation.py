"""
run_evaluation.py
Entry point — runs the full evaluation pipeline.

Usage examples:
  python run_evaluation.py --dataset hotpotqa --n_samples 100
  python run_evaluation.py --dataset hotpotqa --n_samples 200 --compare_baseline
  python run_evaluation.py --dataset ragbench --subset covidqa --n_samples 100
  python run_evaluation.py --dataset ragbench --subset hotpotqa --n_samples 100
  python run_evaluation.py --dataset both --n_samples 100
"""

import argparse
import json
import os
from pipeline import SelfHealingRAGPipeline
from evaluation.evaluator import Evaluator
from utils.helpers import format_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Healing Agentic RAG Evaluation")
    parser.add_argument(
        "--dataset", choices=["hotpotqa", "ragbench", "both"], default="hotpotqa",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--subset", default="hotpotqa",
        help="RAGBench subset (covidqa, hotpotqa, msmarco, pubmedqa, expertqa, ...)"
    )
    parser.add_argument(
        "--setting", choices=["distractor", "fullwiki"], default="distractor",
        help="HotpotQA setting"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--compare_baseline", action="store_true",
        help="Also run baseline (no self-healing) for comparison"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = SelfHealingRAGPipeline()
    evaluator = Evaluator(pipeline)
    all_metrics = []

    # ── HotpotQA Evaluation ────────────────────────────────────────────────────
    if args.dataset in ("hotpotqa", "both"):
        from data.hotpotqa_loader import load_hotpotqa
        samples = load_hotpotqa(n_samples=args.n_samples, setting=args.setting)
        metrics = evaluator.evaluate_hotpotqa(samples, compare_baseline=args.compare_baseline)
        all_metrics.append(("Self-Healing RAG (HotpotQA)", metrics))

        if args.compare_baseline and "baseline_em" in metrics:
            all_metrics.append((
                "Baseline RAG (HotpotQA)",
                {"em": metrics["baseline_em"], "f1": metrics["baseline_f1"],
                 "hallucination_rate": 1.0, "correction_rate": 0.0}
            ))

    # ── RAGBench Evaluation ────────────────────────────────────────────────────
    if args.dataset in ("ragbench", "both"):
        from data.ragbench_loader import load_ragbench
        subsets = (
            ["covidqa", "hotpotqa", "msmarco"]
            if args.dataset == "both"
            else [args.subset]
        )
        for subset in subsets:
            try:
                samples = load_ragbench(subset=subset, split="test", n_samples=args.n_samples)
                metrics = evaluator.evaluate_ragbench(samples, subset=subset)
                all_metrics.append((f"Self-Healing RAG (RAGBench/{subset})", metrics))
            except Exception as e:
                print(f"[Warning] Could not load RAGBench subset '{subset}': {e}")

    # ── Print Comparison Table ─────────────────────────────────────────────────
    if len(all_metrics) > 1:
        print("\n" + format_results_table(all_metrics))

    # ── Save summary ───────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    summary_path = "results/summary.json"
    with open(summary_path, "w") as f:
        json.dump({name: m for name, m in all_metrics}, f, indent=2)
    print(f"\n[Summary] All results saved to {summary_path}")


if __name__ == "__main__":
    main()


# ── Quick preset commands (add to bottom of file for reference) ────────────────
# python run_evaluation.py --dataset hotpotqa --n_samples 200 --compare_baseline
# python run_evaluation.py --dataset ragbench --subset covidqa --n_samples 100
# python run_evaluation.py --dataset ragbench --subset hotpotqa --n_samples 100
# python run_evaluation.py --dataset both --n_samples 100
