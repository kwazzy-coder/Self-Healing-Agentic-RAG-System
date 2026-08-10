"""
evaluation/cost_analysis.py — Cost-Efficiency Analysis

Generates the key differentiating tables and figures for the paper:
compares our tiered approach against always-correct and LLM-judge baselines.
"""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def generate_cost_comparison_table(system_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate LaTeX table comparing cost across systems.
    
    Args:
        system_results: {system_name: {"metrics": {...}, "cost_stats": {...}}}
    
    Returns:
        LaTeX formatted table string.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cost-Efficiency Comparison. Our tiered approach achieves competitive accuracy with significantly fewer LLM calls.}",
        r"\label{tab:cost}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"System & EM & F1 & Avg LLM Calls & Avg Latency (s) \\",
        r"\midrule",
    ]
    
    for name, data in system_results.items():
        metrics = data.get("metrics", {})
        costs = data.get("cost_stats", {})
        em = metrics.get("em", 0)
        f1 = metrics.get("f1", 0)
        llm_calls = costs.get("avg_llm_calls", 0)
        latency = costs.get("avg_latency_seconds", 0)
        
        bold = r"\textbf" if "ours" in name.lower() or "etsc" in name.lower() else ""
        if bold:
            lines.append(f"  {bold}{{{name}}} & {bold}{{{em:.3f}}} & {bold}{{{f1:.3f}}} & {bold}{{{llm_calls:.1f}}} & {bold}{{{latency:.1f}}} \\\\")
        else:
            lines.append(f"  {name} & {em:.3f} & {f1:.3f} & {llm_calls:.1f} & {latency:.1f} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_tier_distribution_table(tier_stats: Dict[str, float]) -> str:
    """
    Generate LaTeX table showing what % of queries resolved at each monitoring tier.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Monitor Tier Distribution. Majority of queries resolved at low-cost tiers.}",
        r"\label{tab:tiers}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Tier & Description & \% Queries \\",
        r"\midrule",
        f"  Tier 1 & Zero-cost heuristics & {tier_stats.get('tier_1_heuristic', 0)*100:.1f}\\% \\\\",
        f"  Tier 2 & Lightweight NLI (67M) & {tier_stats.get('tier_2_nli', 0)*100:.1f}\\% \\\\",
        f"  Tier 3 & LLM-as-judge (1 API call) & {tier_stats.get('tier_3_llm', 0)*100:.1f}\\% \\\\",
        f"  Pass & No correction needed & {tier_stats.get('tier_0', 0)*100:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def compute_cost_savings(our_stats: Dict, baseline_stats: Dict) -> Dict[str, float]:
    """Compute cost savings vs a baseline."""
    our_calls = our_stats.get("avg_llm_calls", 1)
    base_calls = baseline_stats.get("avg_llm_calls", 1)
    our_latency = our_stats.get("avg_latency_seconds", 1)
    base_latency = baseline_stats.get("avg_latency_seconds", 1)
    
    return {
        "llm_call_reduction": (base_calls - our_calls) / base_calls * 100 if base_calls > 0 else 0,
        "latency_reduction": (base_latency - our_latency) / base_latency * 100 if base_latency > 0 else 0,
        "cost_ratio": our_calls / base_calls if base_calls > 0 else 1.0,
    }
