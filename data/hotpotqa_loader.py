"""
data/hotpotqa_loader.py — HotpotQA (Yang et al., 2018), distractor setting.

Each question comes with 10 paragraphs: 2 gold "supporting" paragraphs and
8 distractors, which makes it a good test of retrieval + multi-hop reasoning.

Returned schema: id, question, answer, context_docs [{title, text}],
supporting_titles, type ("bridge" | "comparison"), level.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
from data._hf import load_first_available, seeded_sample

logger = logging.getLogger(__name__)

HUB_IDS = ("hotpotqa/hotpot_qa", "hotpot_qa")


def load_hotpotqa(
    n_samples: Optional[int] = None,
    setting: str = "distractor",
    seed: Optional[int] = None,
    question_type: Optional[str] = None,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Args:
        n_samples: random subset size (None = full split).
        setting: "distractor" or "fullwiki".
        seed: sampling seed (defaults to config.RANDOM_SEED).
        question_type: optional filter, "bridge" or "comparison".
    """
    split = split or config.HOTPOTQA_SPLIT
    logger.info("Loading HotpotQA (%s, %s)...", setting, split)
    ds = load_first_available(HUB_IDS, setting, split=split)

    samples = []
    for row in ds:
        if question_type and row["type"] != question_type:
            continue
        context_docs = [
            {"title": title, "text": " ".join(sents)}
            for title, sents in zip(row["context"]["title"], row["context"]["sentences"], strict=False)
        ]
        samples.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "context_docs": context_docs,
            "supporting_titles": sorted(set(row["supporting_facts"]["title"])),
            "type": row["type"],
            "level": row.get("level", ""),
        })
    samples = seeded_sample(samples, n_samples, config.RANDOM_SEED if seed is None else seed)
    logger.info("Loaded %d HotpotQA samples.", len(samples))
    return samples


def format_context_for_retrieval(context_docs: list) -> list:
    """[{title, text}] -> ["Title: <title>\\n<text>", ...]"""
    return [f"Title: {d['title']}\n{d['text']}" for d in context_docs]
