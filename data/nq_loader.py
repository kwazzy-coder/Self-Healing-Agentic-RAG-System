"""
data/nq_loader.py — Natural Questions Open loader
Single-hop QA — tests whether our system doesn't hurt simple queries.
"""
from datasets import load_dataset
import random
import logging
from typing import List, Dict, Any

try:
    from config import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42

logger = logging.getLogger(__name__)

def load_nq(n_samples: int = 100, split: str = "validation") -> List[Dict[str, Any]]:
    """
    Load the Natural Questions Open dataset.
    
    Args:
        n_samples: Number of samples to load.
        split: Dataset split to load.
        
    Returns:
        List of formatted dataset samples.
    """
    try:
        ds = load_dataset("nq_open", split=split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load nq_open dataset: {e}")
        return []
        
    samples = []
    for i, row in enumerate(ds):
        answer = row.get("answer", [])
        if isinstance(answer, list) and len(answer) > 0:
            answer = answer[0]
            
        samples.append({
            "id": f"nq_{i}",
            "question": row["question"],
            "answer": str(answer),
            "context_docs": [],
            "supporting_titles": [],
            "type": "single-hop"
        })
        
    if n_samples and n_samples < len(samples):
        random.seed(RANDOM_SEED)
        samples = random.sample(samples, n_samples)
    return samples
