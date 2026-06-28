"""
retrieval/retriever.py
Hybrid retriever: BM25 (sparse) + FAISS (dense), score-fused.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from retrieval.vector_store import VectorStore
from config import TOP_K_RETRIEVAL, BM25_WEIGHT, DENSE_WEIGHT


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    """
    Combines BM25 and dense retrieval via linear score fusion.
    For HotpotQA: index = the 10 distractor paragraphs per question.
    For RAGBench: index = the 4 provided documents per question.
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25 = None
        self.documents = []

    def index_documents(self, documents: list[str]):
        """
        Build BM25 + FAISS index over the given documents.
        Call this fresh for each new sample.
        """
        self.documents = documents
        tokenized = [_tokenize(d) for d in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.vector_store.build_index(documents)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """
        Retrieve top-k documents via hybrid scoring.

        Returns:
            List of {"text": str, "hybrid_score": float, "dense_score": float, "bm25_score": float}
        """
        n = len(self.documents)
        if n == 0:
            return []

        # ── BM25 scores ───────────────────────────────────────────────────────
        bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)), dtype=float)
        bm25_max = bm25_scores.max() + 1e-9
        bm25_norm = bm25_scores / bm25_max

        # ── Dense scores ──────────────────────────────────────────────────────
        dense_results = self.vector_store.search(query, top_k=n)
        dense_scores = np.zeros(n, dtype=float)
        for r in dense_results:
            dense_scores[r["idx"]] = r["score"]
        # dense scores are already cosine in [0,1] on normalised vecs

        # ── Hybrid fusion ─────────────────────────────────────────────────────
        hybrid_scores = BM25_WEIGHT * bm25_norm + DENSE_WEIGHT * dense_scores

        # ── Sort & return top-k ───────────────────────────────────────────────
        ranked = np.argsort(-hybrid_scores)[:top_k]
        results = []
        for idx in ranked:
            results.append({
                "text": self.documents[idx],
                "hybrid_score": float(hybrid_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(bm25_norm[idx]),
                "idx": int(idx),
            })
        return results
