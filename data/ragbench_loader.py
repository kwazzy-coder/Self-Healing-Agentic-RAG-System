"""
data/ragbench_loader.py — RAGBench (Friel et al., 2024; galileo-ai/ragbench).

Each row has a question, the retrieved documents, a *response* written by an
LLM, and human/GPT-4 annotations of whether that response is grounded
(`adherence_score`) plus which sentences are unsupported. That makes it a
labelled benchmark for hallucination *detectors* — exactly what the
self-monitor is.

Subsets: covidqa, cuad, delucionqa, emanual, expertqa, finqa, hagrid,
hotpotqa, msmarco, pubmedqa, tatqa, techqa
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
from data._hf import load_first_available, seeded_sample

logger = logging.getLogger(__name__)


def load_ragbench(
    subset: str = "hotpotqa",
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    logger.info("Loading RAGBench subset=%s split=%s...", subset, split)
    ds = load_first_available(("galileo-ai/ragbench", "rungalileo/ragbench"), subset, split=split)
    samples = []
    for row in ds:
        if row.get("adherence_score") is None:
            continue  # unlabeled rows are useless for detector evaluation
        samples.append({
            "id": str(row.get("id", "")),
            "question": row["question"],
            "documents": list(row["documents"]),
            "gold_response": row["response"],
            "adherence_score": bool(row["adherence_score"]),
            "faithfulness": row.get("ragas_faithfulness"),
            "groundedness": row.get("trulens_groundedness"),
            "relevance_score": row.get("relevance_score"),
            "utilization_score": row.get("utilization_score"),
            "completeness_score": row.get("completeness_score"),
            "unsupported_sentence_keys": row.get("unsupported_response_sentence_keys", []),
        })
    samples = seeded_sample(samples, n_samples, config.RANDOM_SEED if seed is None else seed)
    logger.info("Loaded %d RAGBench/%s samples (%d hallucinated).", len(samples), subset,
                sum(not s["adherence_score"] for s in samples))
    return samples


def get_ragbench_hallucinated_samples(subset: str = "covidqa", split: str = "test") -> List[Dict[str, Any]]:
    """Only the rows whose response is NOT fully grounded."""
    return [s for s in load_ragbench(subset=subset, split=split) if not s["adherence_score"]]
