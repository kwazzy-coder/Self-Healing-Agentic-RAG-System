"""
data/popqa_loader.py — PopQA loader
Long-tail entity QA — tests retrieval quality on rare entities.
"""
from datasets import load_dataset
import random
import logging
import ast

try:
    from config import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42

logger = logging.getLogger(__name__)

def load_popqa(n_samples=100, split="test"):
    """
    Load the PopQA dataset.
    
    Args:
        n_samples: Number of samples to load.
        split: Dataset split to load.
        
    Returns:
        List of formatted dataset samples.
    """
    try:
        ds = load_dataset("akariasai/PopQA", split=split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load PopQA dataset: {e}")
        return []
        
    samples = []
    for row in ds:
        possible_answers = row.get("possible_answers", [])
        answer = row.get("obj", "")
        
        if isinstance(possible_answers, list) and possible_answers:
            answer = possible_answers[0]
        elif isinstance(possible_answers, str):
            try:
                parsed = ast.literal_eval(possible_answers)
                if parsed and isinstance(parsed, list):
                    answer = parsed[0]
            except (ValueError, SyntaxError):
                answer = possible_answers
                
        samples.append({
            "id": str(row.get("id", "")),
            "question": row.get("question", ""),
            "answer": str(answer),
            "context_docs": [],
            "supporting_titles": [],
            "type": "long-tail"
        })
        
    if n_samples and n_samples < len(samples):
        random.seed(RANDOM_SEED)
        samples = random.sample(samples, n_samples)
    return samples
