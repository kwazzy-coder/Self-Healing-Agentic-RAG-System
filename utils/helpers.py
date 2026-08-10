"""
utils/helpers.py
Shared utilities: text processing, normalization, logging, and environment info.
"""

import re
import string
import time
import functools
import logging
import hashlib
import platform
import sys
from typing import List, Callable, Any, Dict

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Configures basic logging.
    
    Args:
        level: Logging level as a string (e.g., 'INFO', 'DEBUG').
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
        
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True
    )
    logger.info(f"Logging initialized at {level} level.")


def clean_text(text: str) -> str:
    """
    Removes extra whitespace and newlines from text.
    
    Args:
        text: Input string.
        
    Returns:
        Cleaned string.
    """
    return " ".join(text.split())


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncates text for display or prompt usage.
    
    Args:
        text: Input string.
        max_chars: Maximum characters to retain before truncating.
        
    Returns:
        Truncated string with trailing ellipsis if needed.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def deduplicate_docs(docs: List[str]) -> List[str]:
    """
    Removes duplicate passages based on full text SHA-256 hash.
    
    Args:
        docs: List of document strings.
        
    Returns:
        Deduplicated list of document strings.
    """
    seen = set()
    result = []
    for d in docs:
        cleaned = d.strip().lower()
        key = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def extract_title_from_doc(doc: str) -> str:
    """
    Extracts the title from passages formatted as 'Title: X\n...'.
    
    Args:
        doc: The input document string.
        
    Returns:
        The extracted title, or an empty string if not found.
    """
    match = re.match(r"Title:\s*(.+?)(?:\n|$)", doc)
    return match.group(1).strip() if match else ""


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution and log at INFO level.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"Timer: {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def get_system_info() -> Dict[str, str]:
    """
    Retrieves system environment info for reproducibility tracking.
    
    Returns:
        A dictionary containing Python version, OS info, and library paths.
    """
    info = {
        "python_version": sys.version.replace('\n', ' '),
        "os_platform": platform.platform(),
        "os_release": platform.release()
    }
    return info
