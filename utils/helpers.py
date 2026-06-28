"""
utils/helpers.py
Shared utilities: text processing, normalization, logging.
"""

import re
import string
import time
import functools
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove extra whitespace and newlines."""
    return " ".join(text.split())


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text for display/prompt use."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def deduplicate_docs(docs: list[str]) -> list[str]:
    """Remove duplicate passages (exact match)."""
    seen = set()
    result = []
    for d in docs:
        key = d.strip().lower()[:100]
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def extract_title_from_doc(doc: str) -> str:
    """Extract title from 'Title: X\n...' formatted passages."""
    match = re.match(r"Title:\s*(.+?)(?:\n|$)", doc)
    return match.group(1).strip() if match else ""


def timer(func):
    """Decorator to time function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def format_results_table(metrics_list: list[tuple[str, dict]]) -> str:
    """
    Format a list of (system_name, metrics_dict) into a readable comparison table.
    """
    if not metrics_list:
        return ""

    headers = ["System", "EM", "F1", "Halluc. Rate", "Correction Rate"]
    rows = []
    for name, m in metrics_list:
        rows.append([
            name,
            f"{m.get('em', 0):.3f}",
            f"{m.get('f1', 0):.3f}",
            f"{m.get('hallucination_rate', 0):.3f}",
            f"{m.get('correction_rate', 0):.3f}",
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_row, sep]
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(row, col_widths)) + " |")
    lines.append(sep)

    return "\n".join(lines)
