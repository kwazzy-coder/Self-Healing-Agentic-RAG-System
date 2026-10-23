"""
data/musique_loader.py — MuSiQue (Trivedi et al., 2022): 2-4 hop questions
composed from single-hop questions, with 20 paragraphs each (harder than HotpotQA).

Same schema as the HotpotQA loader, so the evaluator can use either.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
from data._hf import load_first_available, seeded_sample

logger = logging.getLogger(__name__)

HUB_IDS = ("dgslibisey/MuSiQue", "bdsaglam/musique", "drt/musique")


def load_musique(n_samples: Optional[int] = 100, split: str = "validation",
                 seed: Optional[int] = None, answerable_only: bool = True) -> List[Dict[str, Any]]:
    ds = load_first_available(HUB_IDS, split=split)
    samples = []
    for row in ds:
        if answerable_only and row.get("answerable") is False:
            continue
        paragraphs = row.get("paragraphs", []) or []
        hops = len(row.get("question_decomposition", []) or []) or 2
        samples.append({
            "id": row.get("id", ""),
            "question": row["question"],
            "answer": row.get("answer", ""),
            "answer_aliases": row.get("answer_aliases", []) or [],
            "context_docs": [{"title": p.get("title", ""), "text": p.get("paragraph_text", "")} for p in paragraphs],
            "supporting_titles": sorted({p.get("title", "") for p in paragraphs if p.get("is_supporting")}),
            "type": f"{hops}-hop",
        })
    samples = seeded_sample(samples, n_samples, config.RANDOM_SEED if seed is None else seed)
    logger.info("Loaded %d MuSiQue samples.", len(samples))
    return samples
