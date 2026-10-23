"""
experiments/runner.py — Run one YAML-defined experiment over one or more seeds.

    python -m experiments.runner --config experiments/configs/full_system.yaml
    python -m experiments.runner --config experiments/configs/ablation_no_nli.yaml --dry_run

All experiments on the same dataset/size/seed share one results directory, so
the reference vanilla_rag run is computed once (with default settings) and
reused by every ablation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List

import numpy as np

import config
from experiments.config_schema import ExperimentConfig

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


@contextmanager
def config_overrides(overrides: Dict[str, Any]) -> Iterator[None]:
    """Temporarily set config.py attributes (modules read config.X at call time, so this works)."""
    old = {k: getattr(config, k) for k in overrides if hasattr(config, k)}
    unknown = [k for k in overrides if not hasattr(config, k)]
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}")
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def get_system_metadata() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def pipeline_config_for(exp: ExperimentConfig):
    from pipeline import PipelineConfig

    r, h = exp.retrieval, exp.healing
    return PipelineConfig(
        name=exp.name,
        top_k=r.top_k,
        fusion_method=r.fusion_method,
        max_correction_rounds=h.max_correction_attempts,
        enable_healing=h.enabled,
        enable_planning=r.enabled,
        monitor_mode=h.monitor_mode,
        always_correct=h.always_correct,
        use_retrieval=r.enabled,
    )


def load_dataset_samples(exp: ExperimentConfig, seed: int) -> List[Dict[str, Any]]:
    d = exp.dataset
    if d.name == "hotpotqa":
        from data.hotpotqa_loader import load_hotpotqa

        return load_hotpotqa(d.n_samples, seed=seed, question_type=d.question_type)
    if d.name == "musique":
        from data.musique_loader import load_musique

        return load_musique(d.n_samples, seed=seed)
    raise ValueError(f"Unsupported dataset for QA experiments: {d.name}")


def run_experiment(config_path: str, output_dir: str = "results/experiments", dry_run: bool = False,
                   components: Any = None, samples_override: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    exp = ExperimentConfig.from_yaml(config_path)
    logger.info("Experiment '%s': %s", exp.name, exp.description)
    if dry_run:
        logger.info("Config OK. Overrides: %s | pipeline: %s", exp.config_overrides(), pipeline_config_for(exp))
        return {"experiment_name": exp.name, "dry_run": True}

    from agent.llm_reasoner import LLMReasoner
    from baselines import build_system
    from evaluation.evaluator import Evaluator
    from pipeline import Components, SelfHealingRAGPipeline

    if components is None:
        with config_overrides({"LLM_MODEL": exp.llm.model_name} if exp.llm.model_name else {}):
            components = Components.build(llm=LLMReasoner(exp.llm.backend))
    evaluator = Evaluator(results_dir=output_dir)
    per_seed = []
    for seed in exp.seeds:
        set_seeds(seed)
        samples = samples_override or load_dataset_samples(exp, seed)
        tag = f"{exp.dataset.name}{'-' + exp.dataset.question_type if exp.dataset.question_type else ''}" \
              f"_n{exp.dataset.n_samples}/seed{seed}"
        if exp.compare_baseline:  # reference run with default settings, shared by all experiments
            evaluator.evaluate_qa({"vanilla_rag": build_system("vanilla_rag", components)}, samples, tag,
                                  summary_name="summary_vanilla_rag")
        with config_overrides(exp.config_overrides()):
            systems = {exp.name: SelfHealingRAGPipeline(pipeline_config_for(exp), components=components)}
            if exp.compare_baseline:
                systems = {"vanilla_rag": build_system("vanilla_rag", components), **systems}
            summary = evaluator.evaluate_qa(systems, samples, tag, summary_name=f"summary_{exp.name}",
                                            meta={"experiment": exp.name, "seed": seed})
        per_seed.append({"seed": seed, "summary": summary})

    from evaluation.statistical import aggregate_across_seeds

    agg = {}
    for system in per_seed[0]["summary"]["systems"]:
        agg[system] = aggregate_across_seeds([s["summary"]["systems"][system] for s in per_seed])
    report = {
        "experiment_name": exp.name,
        "description": exp.description,
        "config": exp.model_dump(),
        "metadata": get_system_metadata(),
        "seeds": exp.seeds,
        "aggregate": agg,
        "comparisons_by_seed": {s["seed"]: s["summary"]["comparisons"] for s in per_seed},
    }
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{exp.name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Saved %s", out)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one experiment config")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="results/experiments")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_experiment(args.config, args.out_dir, args.dry_run)


if __name__ == "__main__":
    main()
