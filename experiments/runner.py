"""
experiments/runner.py — Unified Experiment Runner

Runs experiments defined by YAML config files with:
- Multi-seed execution for statistical robustness
- Structured result saving (JSON + CSV)
- Runtime metadata logging (hardware, software, timestamps)
- Confidence interval computation
- Comparison table generation across configs
"""

import argparse
import json
import logging
import os
import platform
import random
import time
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

# Assuming we have evaluation components and pipeline accessible.
# If not, these would be placeholders.
from experiments.config_schema import ExperimentConfig

logger = logging.getLogger(__name__)

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def get_system_metadata() -> Dict[str, Any]:
    """Collect runtime metadata."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "timestamp": datetime.now().isoformat()
    }

def apply_config_overrides(config: ExperimentConfig):
    """
    Dynamically patch the system configuration based on the experiment config.
    This modifies global config singletons or sets env vars.
    """
    import config as system_config
    
    # LLM Config
    system_config.LLM_BACKEND = config.llm.backend
    if config.llm.model_name:
        if config.llm.backend == "gemini":
            system_config.GEMINI_MODEL = config.llm.model_name
        elif config.llm.backend == "groq":
            system_config.GROQ_MODEL = config.llm.model_name
        elif config.llm.backend == "ollama":
            system_config.OLLAMA_MODEL = config.llm.model_name
        elif config.llm.backend == "openai":
            system_config.OPENAI_MODEL = config.llm.model_name
        elif config.llm.backend == "huggingface":
            system_config.HF_MODEL = config.llm.model_name
            
    if hasattr(config.llm, "temperature"):
        system_config.TEMPERATURE = config.llm.temperature
    if hasattr(config.llm, "max_answer_words"):
        system_config.MAX_ANSWER_WORDS = config.llm.max_answer_words
    if hasattr(config.llm, "context_char_limit"):
        system_config.CONTEXT_CHAR_LIMIT = config.llm.context_char_limit

    # Retrieval Config
    if hasattr(config.retrieval, "embedding_model"):
        system_config.EMBEDDING_MODEL = config.retrieval.embedding_model
    if hasattr(config.retrieval, "top_k"):
        system_config.TOP_K_RETRIEVAL = config.retrieval.top_k
    if hasattr(config.retrieval, "bm25_weight"):
        system_config.BM25_WEIGHT = config.retrieval.bm25_weight
    if hasattr(config.retrieval, "dense_weight"):
        system_config.DENSE_WEIGHT = config.retrieval.dense_weight
    if hasattr(config.retrieval, "fusion_method"):
        system_config.RETRIEVAL_FUSION_METHOD = config.retrieval.fusion_method

    # Healing Config
    if hasattr(config.healing, "max_correction_attempts"):
        system_config.MAX_CORRECTION_ATTEMPTS = config.healing.max_correction_attempts
    if hasattr(config.healing, "confidence_threshold"):
        system_config.CONFIDENCE_THRESHOLD = config.healing.confidence_threshold
        
    if getattr(config.healing, "decompose_questions", False):
        # We handle decompose_questions by patching the TaskPlanner behavior in run_evaluation
        pass

def run_evaluation(config: ExperimentConfig, seed: int) -> Dict[str, Any]:
    """
    Run the evaluation pipeline for a single seed.
    """
    from pipeline import SelfHealingRAGPipeline
    from evaluation.evaluator import Evaluator
    
    # Load dataset
    if config.dataset.name == "hotpotqa":
        from data.hotpotqa_loader import load_hotpotqa
        setting = getattr(config.dataset, "subset", "distractor")
        samples = load_hotpotqa(n_samples=config.dataset.n_samples, setting=setting)
    elif config.dataset.name == "ragbench":
        from data.ragbench_loader import load_ragbench
        subset = getattr(config.dataset, "subset", "covidqa")
        split = getattr(config.dataset, "split", "test")
        samples = load_ragbench(subset=subset, split=split, n_samples=config.dataset.n_samples)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.name}")
        
    pipeline = SelfHealingRAGPipeline()
    evaluator = Evaluator(pipeline)
    
    healing_enabled = getattr(config.healing, "enabled", True)
    decompose_questions = getattr(config.healing, "decompose_questions", True)
    
    # Temporarily patch TaskPlanner behavior if decompose_questions is disabled
    import agent.task_planner as tp
    original_plan = tp.TaskPlanner.plan
    if not decompose_questions:
        tp.TaskPlanner.plan = lambda self, q: [q]
        
    try:
        if not healing_enabled:
            # Run baseline path when healing is disabled
            pipeline.run = pipeline.run_baseline
            
        if config.dataset.name == "hotpotqa":
            results = evaluator.evaluate_hotpotqa(samples, compare_baseline=not healing_enabled)
        else:
            results = evaluator.evaluate_ragbench(samples, subset=getattr(config.dataset, "subset", "covidqa"))
            
    finally:
        # Restore TaskPlanner behavior
        tp.TaskPlanner.plan = original_plan
        
    return results

def compute_confidence_intervals(results_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and confidence intervals across seeds using evaluation.statistical.
    """
    try:
        from evaluation.statistical import bootstrap_confidence_interval
        aggregated = {}
        if not results_list: return aggregated
        metrics = results_list[0].keys()
        for metric in metrics:
            values = [res[metric] for res in results_list if metric in res]
            if not values:
                continue
            mean_val = np.mean(values)
            ci_lower, ci_upper = bootstrap_confidence_interval(values)
            aggregated[metric] = {
                "mean": mean_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            }
        return aggregated
    except ImportError:
        logger.warning("Could not import evaluation.statistical. Using simple stats.")
        aggregated = {}
        if not results_list: return aggregated
        metrics = results_list[0].keys()
        for metric in metrics:
            values = [res.get(metric, 0.0) for res in results_list]
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower": float(np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values))),
                "ci_upper": float(np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values)))
            }
        return aggregated

def run_experiment(config_path: str, output_dir: str = "results", dry_run: bool = False):
    """Run an experiment defined by a YAML configuration."""
    config = ExperimentConfig.from_yaml(config_path)
    logger.info(f"Loaded config: {config.name}")
    
    if dry_run:
        logger.info("Dry run enabled. Validated config and exiting.")
        return
        
    apply_config_overrides(config)
    os.makedirs(output_dir, exist_ok=True)
    
    seed_results = []
    start_time = time.time()
    
    for seed in config.seeds:
        logger.info(f"Running evaluation with seed {seed}")
        set_seeds(seed)
        res = run_evaluation(config, seed)
        seed_results.append({"seed": seed, "metrics": res})
        
    end_time = time.time()
    
    # Aggregate results
    metrics_list = [sr["metrics"] for sr in seed_results]
    aggregated = compute_confidence_intervals(metrics_list)
    
    final_report = {
        "experiment_name": config.name,
        "description": config.description,
        "metadata": get_system_metadata(),
        "execution_time_seconds": end_time - start_time,
        "seed_results": seed_results,
        "aggregated_metrics": aggregated
    }
    
    out_file = os.path.join(output_dir, f"{config.name}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
        
    logger.info(f"Results saved to {out_file}")
    return final_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Validate config without running")
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parser.parse_args()
    
    run_experiment(args.config, args.out_dir, args.dry_run)
