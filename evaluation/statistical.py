"""
evaluation/statistical.py — Confidence intervals and paired significance tests.

* bootstrap_confidence_interval — percentile bootstrap CI of a mean
* paired_bootstrap_test         — two systems on the same questions (continuous metric, e.g. F1)
* mcnemar_test                  — two systems on the same questions (binary metric, e.g. EM)

With 100-200 questions, a 2-3 point EM difference is often *not* significant;
reporting p-values keeps the claims honest.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def bootstrap_confidence_interval(
    scores: Sequence[float], n_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 0
) -> Tuple[float, float, float]:
    """Returns (mean, ci_lower, ci_upper)."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_bootstrap, arr.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2
    return mean, float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_bootstrap_test(
    scores_a: Sequence[float], scores_b: Sequence[float], n_bootstrap: int = 5000, seed: int = 0
) -> Dict[str, Any]:
    """Two-sided paired bootstrap test of mean(a) - mean(b) = 0."""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Score lists must have the same length for a paired test.")
    if a.size == 0:
        return {"mean_diff": 0.0, "p_value": 1.0, "effect_size": 0.0, "significant": False}
    diffs = a - b
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    centred = diffs - observed
    boot = centred[rng.integers(0, diffs.size, size=(n_bootstrap, diffs.size))].mean(axis=1)
    p_val = float(min(1.0, (np.abs(boot) >= abs(observed)).mean()))
    if observed == 0.0:
        p_val = 1.0
    return {
        "mean_diff": observed,
        "p_value": p_val,
        "effect_size": cohens_d(a, b),
        "significant": p_val < 0.05,
    }


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, Any]:
    """McNemar's test on paired binary outcomes (exact binomial when discordant pairs < 25)."""
    if len(correct_a) != len(correct_b):
        raise ValueError("Lists must have the same length.")
    b = sum(1 for x, y in zip(correct_a, correct_b, strict=False) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=False) if not x and y)
    if b + c == 0:
        return {"p_value": 1.0, "statistic": 0.0, "a_only": b, "b_only": c, "significant": False}
    if b + c < 25:
        p_val = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        stat = float(min(b, c))
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_val = float(stats.chi2.sf(stat, 1))
    return {"p_value": float(p_val), "statistic": float(stat), "a_only": b, "b_only": c,
            "significant": p_val < 0.05}


def cohens_d(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Cohen's d for paired samples (mean difference / SD of differences)."""
    diffs = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if diffs.size < 2:
        return 0.0
    sd = diffs.std(ddof=1)
    return 0.0 if sd == 0 else float(diffs.mean() / sd)


def aggregate_across_seeds(seed_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """{metric: {mean, std, n}} over the numeric metrics of several runs."""
    if not seed_results:
        return {}
    keys = {k for r in seed_results for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out = {}
    for k in sorted(keys):
        vals = [float(r[k]) for r in seed_results if isinstance(r.get(k), (int, float))]
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def format_with_ci(mean: float, ci_lower: float, ci_upper: float) -> str:
    """'0.756 [0.712, 0.801]'"""
    return f"{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
