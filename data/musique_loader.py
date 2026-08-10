"""
data/musique_loader.py — MuSiQue dataset loader
MuSiQue: Multi-hop Questions via Single-hop Question Composition
Harder than HotpotQA — supports 2, 3, and 4-hop reasoning.
"""
from datasets import load_dataset
import random
import logging

try:
    from config import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42

logger = logging.getLogger(__name__)

def load_musique(n_samples=100, split="validation"):
    """
    Load the MuSiQue dataset.
    
    Args:
        n_samples: Number of samples to load.
        split: Dataset split to load.
        
    Returns:
        List of formatted dataset samples.
    """
    try:
        ds = load_dataset("drt/musique", split=split, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load drt/musique: {e}. Trying musique...")
        try:
            ds = load_dataset("musique", split=split, trust_remote_code=True)
        except Exception as e2:
            logger.error(f"Failed to load musique dataset: {e2}")
            return []
            
    samples = []
    for row in ds:
        paragraphs = row.get("paragraphs", [])
        context_docs = []
        supporting_titles = []
        for p in paragraphs:
            context_docs.append({"title": p.get("title", ""), "text": p.get("paragraph_text", "")})
            if p.get("is_supporting", False):
                supporting_titles.append(p.get("title", ""))
        
        samples.append({
            "id": row.get("id", ""),
            "question": row["question"],
            "answer": row.get("answer", ""),
            "context_docs": context_docs,
            "supporting_titles": list(set(supporting_titles)),
            "type": f"{row.get('num_hops', 2)}-hop",
        })
    
    if n_samples and n_samples < len(samples):
        random.seed(RANDOM_SEED)
        samples = random.sample(samples, n_samples)
    return samples
