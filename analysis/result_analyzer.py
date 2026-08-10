"""
analysis/result_analyzer.py — Result Aggregation & LaTeX Table Generation

Loads experiment results, computes comparisons, generates LaTeX-formatted
tables for direct inclusion in research papers.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional

class ResultAnalyzer:
    """Analyzes experimental results and generates LaTeX tables."""

    def __init__(self):
        """Initializes the ResultAnalyzer."""
        pass

    def load_results(self, results_dir: str) -> Dict[str, Any]:
        """
        Load all JSON result files from a directory.

        Args:
            results_dir: Path to directory containing JSON result files.

        Returns:
            Dict mapping filenames to parsed JSON data.
        """
        results = {}
        if not os.path.exists(results_dir):
            return results
            
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        results[filename] = json.load(f)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        return results

    def load_experiment_results(self, experiment_dir: str) -> Dict[str, Any]:
        """
        Load structured experiment results.

        Args:
            experiment_dir: Path to the experiment directory.

        Returns:
            Dict containing aggregated experiment results.
        """
        return self.load_results(experiment_dir)

    def _format_metric(self, value: float, std: Optional[float] = None, is_best: bool = False) -> str:
        """Helper to format a metric value for LaTeX."""
        val_str = f"{value * 100:.1f}" if value <= 1.0 else f"{value:.1f}"
        if std is not None:
            std_str = f"{std * 100:.1f}" if std <= 1.0 else f"{std:.1f}"
            res = f"{val_str} \\pm {std_str}"
        else:
            res = val_str
            
        if is_best:
            return f"\\textbf{{{res}}}"
        return res

    def generate_main_results_table(self, results: Dict[str, Any]) -> str:
        """
        Generate LaTeX main results table showing system vs baselines.

        Args:
            results: Dictionary of aggregated results.

        Returns:
            LaTeX formatted table string.
        """
        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "\\textbf{System} & \\textbf{EM} & \\textbf{F1} & \\textbf{ROUGE-L} & \\textbf{Faithfulness} & \\textbf{Supp Facts F1} \\\\",
            "\\midrule"
        ]
        
        # This is a stub implementation. In a real scenario, this would extract
        # specific metrics from the 'results' dictionary and identify the best scores.
        # Adding some dummy data logic to illustrate structure based on keys in results if any.
        for system_name, metrics in results.items():
            if not isinstance(metrics, dict): continue
            
            # Extract basic metrics (falling back to 0.0)
            em = metrics.get('em', 0.0)
            f1 = metrics.get('f1', 0.0)
            rouge_l = metrics.get('rouge_l', 0.0)
            faith = metrics.get('faithfulness', 0.0)
            supp_f1 = metrics.get('supporting_facts_f1', 0.0)
            
            # Format line (simplified, real would compute best per column)
            line = f"{system_name.replace('_', '\\_')} & {self._format_metric(em)} & {self._format_metric(f1)} & {self._format_metric(rouge_l)} & {self._format_metric(faith)} & {self._format_metric(supp_f1)} \\\\"
            lines.append(line)
            
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Main Results comparing the Self-Healing System against Baselines.}",
            "\\label{tab:main_results}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)

    def generate_ablation_table(self, results: Dict[str, Any]) -> str:
        """
        Generate LaTeX table showing ablation study results.

        Args:
            results: Dictionary of ablation results.

        Returns:
            LaTeX formatted table string.
        """
        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Configuration} & \\textbf{EM} & \\textbf{F1} & \\textbf{Faithfulness} \\\\",
            "\\midrule"
        ]
        
        for system_name, metrics in results.items():
            if not isinstance(metrics, dict): continue
            em = metrics.get('em', 0.0)
            f1 = metrics.get('f1', 0.0)
            faith = metrics.get('faithfulness', 0.0)
            line = f"{system_name.replace('_', '\\_')} & {self._format_metric(em)} & {self._format_metric(f1)} & {self._format_metric(faith)} \\\\"
            lines.append(line)
            
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Ablation Study Results.}",
            "\\label{tab:ablation}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)

    def generate_per_dataset_table(self, results: Dict[str, Any]) -> str:
        """
        Generate breakdown by dataset.

        Args:
            results: Results dictionary.

        Returns:
            LaTeX table string.
        """
        return "% Per dataset table (Not fully implemented)\n"

    def generate_significance_table(self, results: Dict[str, Any]) -> str:
        """
        Generate pairwise significance comparison.

        Args:
            results: Results dictionary.

        Returns:
            LaTeX table string.
        """
        return "% Significance table (Not fully implemented)\n"

    def compare_systems(self, system_a_results: Dict, system_b_results: Dict) -> Dict[str, Any]:
        """
        Compare two systems with statistical significance.

        Args:
            system_a_results: Results of system A.
            system_b_results: Results of system B.

        Returns:
            Dictionary with comparison metrics and p-values.
        """
        return {"comparison": "stub", "p_value": 0.05}

    def export_all_tables(self, results: Dict[str, Any], output_dir: str):
        """
        Export all tables to .tex files.

        Args:
            results: Aggregated results dict.
            output_dir: Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        main_tab = self.generate_main_results_table(results)
        with open(os.path.join(output_dir, "main_results.tex"), "w", encoding="utf-8") as f:
            f.write(main_tab)
            
        ablation_tab = self.generate_ablation_table(results)
        with open(os.path.join(output_dir, "ablation.tex"), "w", encoding="utf-8") as f:
            f.write(ablation_tab)
