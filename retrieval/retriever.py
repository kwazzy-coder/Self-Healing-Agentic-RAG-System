"""
retrieval/retriever.py
Hybrid retriever: BM25 (sparse) + FAISS (dense), score-fused.
"""

import string
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from retrieval.vector_store import VectorStore
from config import TOP_K_RETRIEVAL, BM25_WEIGHT, DENSE_WEIGHT

try:
    from config import RETRIEVAL_FUSION_METHOD
except ImportError:
    RETRIEVAL_FUSION_METHOD = "linear"

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "when", "where", "how", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "of", "in", "for",
    "on", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def _tokenize(text: str) -> List[str]:
    """
    Tokenizes text by lowercasing, removing punctuation, and filtering out stopwords.
    
    Args:
        text: The input string.
        
    Returns:
        List of cleaned tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

@dataclass
class RetrievalResult:
    """Dataclass representing a retrieved document and its scores."""
    text: str
    hybrid_score: float
    dense_score: float
    bm25_score: float
    idx: int

class HybridRetriever:
    """
    Combines BM25 and dense retrieval via linear score fusion or RRF.
    """

    def __init__(self) -> None:
        """Initializes the HybridRetriever with an empty vector store."""
        self.vector_store = VectorStore()
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        logger.info("Initialized HybridRetriever")

    def index_documents(self, documents: List[str]) -> None:
        """
        Builds BM25 + FAISS index over the given documents.
        
        Args:
            documents: List of text documents to index.
        """
        logger.info(f"Indexing {len(documents)} documents...")
        self.documents = documents
        tokenized = [_tokenize(d) for d in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.vector_store.build_index(documents)
        logger.info("Finished indexing documents.")

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[RetrievalResult]:
        """
        Retrieves top-k documents via hybrid scoring.
        
        Args:
            query: The search query.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of RetrievalResult objects.
        """
        n = len(self.documents)
        if n == 0:
            logger.warning("No documents indexed. Returning empty results.")
            return []

        # -- BM25 scores --
        bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)), dtype=float)
        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        if bm25_max - bm25_min > 0:
            bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-9)
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        # -- Dense scores --
        dense_results = self.vector_store.search(query, top_k=n)
        dense_scores = np.zeros(n, dtype=float)
        for r in dense_results:
            dense_scores[r["idx"]] = r["score"]

        dense_min = dense_scores.min()
        dense_max = dense_scores.max()
        if dense_max - dense_min > 0:
            dense_norm = (dense_scores - dense_min) / (dense_max - dense_min + 1e-9)
        else:
            dense_norm = np.zeros_like(dense_scores)

        # -- Hybrid fusion --
        if RETRIEVAL_FUSION_METHOD == "rrf":
            # Reciprocal Rank Fusion
            hybrid_scores = np.zeros(n, dtype=float)
            bm25_ranks = np.argsort(-bm25_scores)
            dense_ranks = np.argsort(-dense_scores)
            
            k = 60
            for rank, doc_idx in enumerate(bm25_ranks):
                hybrid_scores[doc_idx] += 1.0 / (k + rank + 1)
            for rank, doc_idx in enumerate(dense_ranks):
                hybrid_scores[doc_idx] += 1.0 / (k + rank + 1)
        else:
            # Linear combination
            hybrid_scores = BM25_WEIGHT * bm25_norm + DENSE_WEIGHT * dense_norm

        # -- Sort & return top-k --
        ranked = np.argsort(-hybrid_scores)[:top_k]
        results = []
        for idx in ranked:
            results.append(RetrievalResult(
                text=self.documents[idx],
                hybrid_score=float(hybrid_scores[idx]),
                dense_score=float(dense_scores[idx]),
                bm25_score=float(bm25_norm[idx]),
                idx=int(idx),
            ))
            
        logger.debug(f"Retrieved {len(results)} results for query: '{query}'")
        return results
