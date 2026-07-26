"""
evaluation/statistical.py — Statistical Significance & Confidence Intervals

Provides bootstrap confidence intervals, paired significance tests,
and multi-seed aggregation for publication-quality evaluation.
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any

def bootstrap_confidence_interval(scores: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes the mean and bootstrap confidence interval for a list of scores.
    Returns (mean, ci_lower, ci_upper).
    """
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    
    if len(scores) < 2:
        return mean, mean, mean
        
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores_array, size=len(scores_array), replace=True)
        bootstrap_means.append(np.mean(sample))
        
    alpha = 1.0 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha / 2.0 * 100)
    ci_upper = np.percentile(bootstrap_means, (1.0 - alpha / 2.0) * 100)
    
    return float(mean), float(ci_lower), float(ci_upper)


def paired_bootstrap_test(scores_a: List[float], scores_b: List[float], n_bootstrap: int = 1000) -> Dict[str, Any]:
    """
    Performs a paired bootstrap hypothesis test comparing two systems.
    Returns dict with p_value, effect_size, and significance flag.
    """
    assert len(scores_a) == len(scores_b), "Score lists must have the same length for paired test."
    n = len(scores_a)
    
    diffs = np.array(scores_a) - np.array(scores_b)
    mean_diff = np.mean(diffs)
    
    # Null hypothesis: mean difference is 0.
    # Center the differences around 0 for the null distribution.
    centered_diffs = diffs - mean_diff
    
    bootstrap_mean_diffs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(centered_diffs, size=n, replace=True)
        bootstrap_mean_diffs.append(np.mean(sample))
        
    # Two-sided p-value
    if mean_diff >= 0:
        p_val = np.mean(np.array(bootstrap_mean_diffs) >= mean_diff) * 2
    else:
        p_val = np.mean(np.array(bootstrap_mean_diffs) <= mean_diff) * 2
        
    p_val = min(float(p_val), 1.0)
    
    eff_size = cohens_d(scores_a, scores_b)
    
    return {
        "p_value": p_val,
        "effect_size": eff_size,
        "significant": p_val < 0.05
    }


def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, Any]:
    """
    McNemar's test for paired dichotomous data (e.g., correct/incorrect).
    """
    assert len(correct_a) == len(correct_b), "Lists must have the same length."
    
    b = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    c = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    
    if b + c == 0:
        return {"p_value": 1.0, "statistic": 0.0, "significant": False}
        
    # Exact binomial test for small b+c
    if b + c < 25:
        p_val = stats.binom_test(min(b, c), b + c, 0.5)
        stat = 0.0
    else:
        # Asymptotic Chi-square test with continuity correction
        stat = ((abs(b - c) - 1)**2) / (b + c)
        p_val = 1.0 - stats.chi2.cdf(stat, 1)
        
    return {
        "p_value": float(p_val),
        "statistic": float(stat),
        "significant": p_val < 0.05
    }


def cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Computes Cohen's d effect size for paired samples.
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    std_diff = np.std(diffs, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diffs) / std_diff)


def aggregate_across_seeds(seed_results: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Aggregates metrics across multiple seeds.
    Returns {metric_name: (mean, std)}.
    """
    if not seed_results:
        return {}
        
    keys = seed_results[0].keys()
    agg = {}
    
    for k in keys:
        vals = [res.get(k, 0.0) for res in seed_results if isinstance(res.get(k), (int, float))]
        if vals:
            agg[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
            
    return agg


def format_with_ci(mean: float, ci_lower: float, ci_upper: float) -> str:
    """
    Formats a metric with its 95% confidence interval.
    Example: "0.756 (±0.023)"
    """
    # Assuming symmetric interval for display, we compute margin
    margin = (ci_upper - ci_lower) / 2.0
    return f"{mean:.3f} (±{margin:.3f})"
