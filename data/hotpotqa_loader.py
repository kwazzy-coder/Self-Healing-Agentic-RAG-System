"""
data/hotpotqa_loader.py
Loads the HotpotQA validation set.

HotpotQA fields used:
  - question        : the multi-hop question
  - answer          : gold answer string
  - context         : list of [title, sentences] pairs (distractor setting)
  - supporting_facts: list of [title, sent_idx] pairs
  - type            : "comparison" or "bridge"
"""

from datasets import load_dataset
import random
from config import HOTPOTQA_SPLIT, RANDOM_SEED


def load_hotpotqa(n_samples: int = None, setting: str = "distractor"):
    """
    Load HotpotQA samples.

    Args:
        n_samples: if None, return full split; else sample randomly
        setting: 'distractor' (10 para) or 'fullwiki'

    Returns:
        List of dicts with keys:
            id, question, answer, context_docs, supporting_titles, type
    """
    print(f"[HotpotQA] Loading split='{HOTPOTQA_SPLIT}'...")
    ds = load_dataset("hotpot_qa", setting, split=HOTPOTQA_SPLIT, trust_remote_code=True)

    samples = []
    for row in ds:
        # Build flat context: list of {"title": str, "text": str}
        context_docs = []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            text = " ".join(sentences)
            context_docs.append({"title": title, "text": text})

        supporting_titles = [f[0] for f in zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
        )]

        samples.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "context_docs": context_docs,          # 10 docs (distractor setting)
            "supporting_titles": supporting_titles, # ground-truth support docs
            "type": row["type"],                    # comparison or bridge
        })

    if n_samples is not None and n_samples < len(samples):
        random.seed(RANDOM_SEED)
        samples = random.sample(samples, n_samples)

    print(f"[HotpotQA] Loaded {len(samples)} samples.")
    return samples


def format_context_for_retrieval(context_docs: list) -> list:
    """
    Convert context_docs into a flat list of passage strings for indexing.
    Each passage = "Title: {title}\n{text}"
    """
    return [f"Title: {d['title']}\n{d['text']}" for d in context_docs]
