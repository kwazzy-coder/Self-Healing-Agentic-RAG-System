"""data/_hf.py — Small helpers shared by the dataset loaders."""
from __future__ import annotations

import logging
import random
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


def load_first_available(candidates: Sequence[str], name: Optional[str] = None, split: str = "validation") -> Any:
    """
    Try several Hub ids for the same dataset (datasets get renamed/moved, and
    `datasets>=3` no longer runs legacy loading scripts).
    """
    from datasets import load_dataset

    errors = []
    for repo in candidates:
        try:
            return load_dataset(repo, name, split=split) if name else load_dataset(repo, split=split)
        except Exception as err:  # noqa: BLE001
            errors.append(f"{repo}: {type(err).__name__}: {err}")
            logger.debug("Could not load %s (%s)", repo, err)
    raise RuntimeError("Could not load any of " + ", ".join(candidates) + "\n" + "\n".join(errors))


def seeded_sample(items: List[Any], n: Optional[int], seed: int) -> List[Any]:
    """Deterministic random subset (each seed gives a different, reproducible subset)."""
    if n is None or n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)
