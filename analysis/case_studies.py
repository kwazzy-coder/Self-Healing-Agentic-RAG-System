"""
analysis/case_studies.py — Qualitative Case Study Generator

Extracts and formats compelling case studies showing the self-healing
loop in action, suitable for inclusion in a research paper.
"""

import os
from typing import List, Dict, Any

class CaseStudyGenerator:
    """Generates formatted case studies from system execution traces."""

    def __init__(self):
        """Initializes the CaseStudyGenerator."""
        pass

    def find_healing_success_cases(self, results: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
        """
        Find cases where healing significantly improved the answer.

        Args:
            results: List of result dicts.
            n: Number of cases.

        Returns:
            List of case dictionaries.
        """
        # Stub implementation
        cases = [r for r in results if r.get("initial_score", 0) < r.get("final_score", 1)]
        return cases[:n]

    def find_healing_failure_cases(self, results: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
        """
        Find cases where healing made things worse.

        Args:
            results: List of result dicts.
            n: Number of cases.

        Returns:
            List of case dictionaries.
        """
        # Stub implementation
        cases = [r for r in results if r.get("initial_score", 1) > r.get("final_score", 0)]
        return cases[:n]

    def find_interesting_traces(self, results: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
        """
        Find traces with multi-round corrections and different strategies used.

        Args:
            results: List of result dicts.
            n: Number of cases.

        Returns:
            List of case dictionaries.
        """
        cases = [r for r in results if len(r.get("correction_rounds", [])) > 1]
        return cases[:n]

    def format_case_study_latex(self, case: Dict[str, Any]) -> str:
        """
        Format as LaTeX figure showing the healing process.

        Args:
            case: A single result dictionary.

        Returns:
            LaTeX string.
        """
        query = case.get("query", "Unknown query").replace("_", "\\_").replace("&", "\\&")
        initial = case.get("initial_answer", "Unknown").replace("_", "\\_").replace("&", "\\&")
        correction = case.get("correction_action", "None").replace("_", "\\_").replace("&", "\\&")
        final = case.get("final_answer", "Unknown").replace("_", "\\_").replace("&", "\\&")

        lines = [
            "\\begin{figure}[h!]",
            "\\centering",
            "\\begin{tcolorbox}[title=Case Study: Self-Healing Process, colback=white, colframe=black!75!white]",
            f"\\textbf{{Query:}} {query} \\\\[0.5em]",
            f"\\textbf{{Initial Answer:}} {initial} \\\\[0.5em]",
            f"\\textbf{{Correction Action:}} {correction} \\\\[0.5em]",
            f"\\textbf{{Final Answer:}} {final}",
            "\\end{tcolorbox}",
            "\\caption{Example of the self-healing loop improving an initial answer.}",
            "\\end{figure}"
        ]
        return "\n".join(lines)

    def format_case_study_markdown(self, case: Dict[str, Any]) -> str:
        """
        Format as Markdown.

        Args:
            case: A single result dictionary.

        Returns:
            Markdown string.
        """
        query = case.get("query", "Unknown query")
        initial = case.get("initial_answer", "Unknown")
        correction = case.get("correction_action", "None")
        final = case.get("final_answer", "Unknown")

        md = f"### Case Study\n\n"
        md += f"**Query:** {query}\n\n"
        md += f"**Initial Answer:** {initial}\n\n"
        md += f"**Correction Action:** {correction}\n\n"
        md += f"**Final Answer:** {final}\n\n"
        md += "---\n"
        return md

    def generate_case_studies_document(self, results: List[Dict[str, Any]], output_path: str, format: str = 'markdown'):
        """
        Generate complete case studies document.

        Args:
            results: List of result dicts.
            output_path: Path to output file.
            format: 'markdown' or 'latex'.
        """
        successes = self.find_healing_success_cases(results)
        
        with open(output_path, "w", encoding="utf-8") as f:
            if format == 'markdown':
                f.write("# Self-Healing Case Studies\n\n")
                f.write("## Successful Healing\n\n")
                for case in successes:
                    f.write(self.format_case_study_markdown(case))
            elif format == 'latex':
                for case in successes:
                    f.write(self.format_case_study_latex(case))
                    f.write("\n\n")
