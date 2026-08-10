"""
experiments/run_all_experiments.py — Run All Ablation Experiments

Sequentially runs all experiment configurations and generates
a comprehensive comparison report.
"""

import argparse
import glob
import json
import logging
import os
import time
from typing import Dict, Any

import pandas as pd

from experiments.runner import run_experiment

logger = logging.getLogger(__name__)

def generate_comparison_table(results_dir: str, output_file: str):
    """Generates a CSV comparison table from all JSON results."""
    result_files = glob.glob(os.path.join(results_dir, "*_results.json"))
    data = []
    
    for fpath in result_files:
        with open(fpath, "r", encoding="utf-8") as f:
            res = json.load(f)
            
        row = {
            "Experiment": res["experiment_name"],
            "Description": res.get("description", "")
        }
        
        # Flatten metrics
        for metric, stats in res.get("aggregated_metrics", {}).items():
            row[f"{metric}_mean"] = stats.get("mean", 0.0)
            row[f"{metric}_ci_lower"] = stats.get("ci_lower", 0.0)
            row[f"{metric}_ci_upper"] = stats.get("ci_upper", 0.0)
            
        data.append(row)
        
    if not data:
        logger.warning("No results found to generate comparison table.")
        return
        
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Comparison table saved to {output_file}")
    
def run_all(configs_dir: str, results_dir: str, force: bool = False):
    """Discovers and runs all YAML configs."""
    os.makedirs(results_dir, exist_ok=True)
    
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
    config_files.sort()
    
    total = len(config_files)
    logger.info(f"Found {total} configurations in {configs_dir}")
    
    start_time = time.time()
    
    for i, cfg_path in enumerate(config_files, 1):
        filename = os.path.basename(cfg_path)
        exp_name = os.path.splitext(filename)[0]
        result_path = os.path.join(results_dir, f"{exp_name}_results.json")
        
        if os.path.exists(result_path) and not force:
            logger.info(f"[{i}/{total}] Skipping {exp_name} (results already exist)")
            continue
            
        logger.info(f"[{i}/{total}] Running {exp_name}...")
        try:
            run_experiment(cfg_path, output_dir=results_dir, dry_run=False)
        except Exception as e:
            logger.error(f"Error running {exp_name}: {str(e)}")
            continue
            
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        eta = avg_time * (total - i)
        logger.info(f"Estimated time remaining: {eta/60:.2f} minutes")
        
    # Generate master comparison table
    compare_file = os.path.join(results_dir, "master_comparison.csv")
    generate_comparison_table(results_dir, compare_file)
    logger.info("All experiments completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments sequentially")
    parser.add_argument("--configs_dir", type=str, default="experiments/configs", help="Directory with YAML configs")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parser.parse_args()
    
    # Need to be relative to project root or use absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs = os.path.join(base_dir, args.configs_dir)
    results = os.path.join(base_dir, args.results_dir)
    
    run_all(configs, results, args.force)
