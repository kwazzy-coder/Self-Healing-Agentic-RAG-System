"""
experiments/run_all_experiments.py — Run every YAML config and build one comparison table.

    python -m experiments.run_all_experiments                 # run (skips finished ones)
    python -m experiments.run_all_experiments --table_only    # just rebuild the table
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import List

from experiments.runner import run_experiment

logger = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def comparison_table(results_dir: str) -> str:
    rows: List[str] = [
        "| Experiment | EM % | F1 % | LLM calls / q | ΔEM vs vanilla (pp) | Description |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        name = res.get("experiment_name")
        agg = res.get("aggregate", {}).get(name, {})
        if not agg:
            continue
        deltas = [c.get(name, {}).get("delta_em") for c in res.get("comparisons_by_seed", {}).values()]
        deltas = [d for d in deltas if d is not None]
        delta = f"{100 * sum(deltas) / len(deltas):+.1f}" if deltas else "–"

        def pct(k: str, agg: dict = agg) -> str:
            v = agg.get(k, {}).get("mean")
            sd = agg.get(k, {}).get("std", 0)
            return "–" if v is None else (f"{100 * v:.1f} ± {100 * sd:.1f}" if sd else f"{100 * v:.1f}")

        calls = agg.get("avg_llm_calls", {}).get("mean")
        rows.append(f"| {name} | {pct('em')} | {pct('f1')} | {'–' if calls is None else f'{calls:.2f}'} | "
                    f"{delta} | {res.get('description', '')} |")
    return "\n".join(rows)


def run_all(configs_dir: str, results_dir: str, force: bool = False) -> None:
    os.makedirs(results_dir, exist_ok=True)
    configs = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    # Run the full system first so its numbers exist even if a later ablation fails.
    configs.sort(key=lambda p: (os.path.basename(p) != "full_system.yaml", p))
    for i, path in enumerate(configs, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(results_dir, f"{name}.json")) and not force:
            logger.info("[%d/%d] %s already done, skipping", i, len(configs), name)
            continue
        logger.info("[%d/%d] running %s", i, len(configs), name)
        try:
            run_experiment(path, output_dir=results_dir)
        except Exception as err:
            logger.error("%s failed: %s", name, err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs_dir", default=os.path.join(ROOT, "experiments", "configs"))
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "results", "experiments"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--table_only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.table_only:
        run_all(args.configs_dir, args.results_dir, args.force)
    table = comparison_table(args.results_dir)
    with open(os.path.join(args.results_dir, "ablations.md"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
