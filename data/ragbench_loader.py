"""
data/ragbench_loader.py
Loads the galileo-ai/ragbench dataset from HuggingFace.

RAGBench fields used:
  - question              : the question
  - documents             : list of 4 passage strings
  - response              : LLM-generated answer (ground truth baseline)
  - adherence_score       : bool — is response fully grounded?
  - faithfulness scores   : ragas_faithfulness, trulens_groundedness, gpt3_adherence
  - relevance_score       : how relevant retrieved docs are
  - sentence_support_info : per-sentence support labels (for fine-grained eval)
  - unsupported_keys      : which response sentences are NOT supported

Available subsets (12 total):
  covidqa, cuad, delucionqa, emanual, expertqa, finqa,
  hagrid, hotpotqa, msmarco, pubmedqa, tatqa, techqa
"""

from datasets import load_dataset
import random
from config import RANDOM_SEED


def load_ragbench(subset: str = "hotpotqa", split: str = "test", n_samples: int = None):
    """
    Load a RAGBench subset.

    Args:
        subset   : one of the 12 available subsets
        split    : 'train', 'validation', or 'test'
        n_samples: if None, return full split

    Returns:
        List of dicts with standardised keys.
    """
    print(f"[RAGBench] Loading subset='{subset}', split='{split}'...")
    ds = load_dataset("galileo-ai/ragbench", subset, split=split, trust_remote_code=True)

    samples = []
    for row in ds:
        samples.append({
            "id": row.get("id", ""),
            "question": row["question"],
            "documents": row["documents"],           # 4 passage strings
            "gold_response": row["response"],        # GPT-3.5 baseline answer
            "adherence_score": row["adherence_score"],   # True/False
            "faithfulness": row.get("ragas_faithfulness", None),
            "groundedness": row.get("trulens_groundedness", None),
            "gpt3_adherence": row.get("gpt3_adherence", None),
            "relevance_score": row.get("relevance_score", None),
            "utilization_score": row.get("utilization_score", None),
            "completeness_score": row.get("completeness_score", None),
            # Sentence-level support: which response sentences are unsupported
            "unsupported_sentence_keys": row.get("unsupported_response_sentence_keys", []),
            "sentence_support_info": row.get("sentence_support_information", []),
        })

    if n_samples is not None and n_samples < len(samples):
        random.seed(RANDOM_SEED)
        samples = random.sample(samples, n_samples)

    print(f"[RAGBench] Loaded {len(samples)} samples from '{subset}'.")
    return samples


def get_ragbench_hallucinated_samples(subset: str = "covidqa", split: str = "test"):
    """
    Return only samples where adherence_score=False (the LLM hallucinated).
    Useful for testing the self-monitoring module specifically.
    """
    all_samples = load_ragbench(subset=subset, split=split)
    hallucinated = [s for s in all_samples if not s["adherence_score"]]
    print(f"[RAGBench] {len(hallucinated)}/{len(all_samples)} samples are hallucinated in '{subset}'.")
    return hallucinated
