"""
retrieval/vector_store.py
FAISS-based dense vector store for document retrieval.
Uses sentence-transformers for embeddings.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TOP_K_RETRIEVAL


class VectorStore:
    """
    Lightweight in-memory FAISS index.
    One instance is created per query (documents change each time for
    the distractor/fullwiki settings of HotpotQA).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"[VectorStore] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents: list[str]):
        """
        Index a list of passage strings.

        Args:
            documents: list of plain text passages
        """
        self.documents = documents
        embeddings = self.model.encode(documents, show_progress_bar=False, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalised vecs
        self.index.add(embeddings.astype("float32"))

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """
        Return top-k passages for a query.

        Returns:
            List of {"text": str, "score": float, "idx": int}
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "text": self.documents[idx],
                    "score": float(score),
                    "idx": int(idx),
                })
        return results
