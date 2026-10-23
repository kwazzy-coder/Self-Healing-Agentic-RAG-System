"""
retrieval/retriever.py — Hybrid sparse + dense retrieval.

BM25 (lexical) and a sentence-embedding index (semantic) are fused either by
a weighted sum of min-max-normalised scores ("linear") or by Reciprocal Rank
Fusion ("rrf"). BM25 catches exact entity names; dense retrieval catches
paraphrases — multi-hop questions need both.

``HybridRetriever.build_index(docs)`` returns an immutable ``HybridIndex``.
Nothing is stored on the retriever itself, so one retriever can safely serve
concurrent API requests.
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

import config
from retrieval.vector_store import DenseIndex, VectorStore

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "when", "where", "how", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "which",
    "have", "has", "had", "do", "does", "did", "to", "of", "in", "for",
    "on", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
}
_PUNCT = str.maketrans("", "", string.punctuation)
_TITLE_RE = re.compile(r"^Title:\s*(.+?)(?:\n|$)")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords."""
    return [t for t in text.lower().translate(_PUNCT).split() if t not in STOPWORDS]


# Backwards-compatible alias
_tokenize = tokenize


def extract_title(doc: str) -> str:
    m = _TITLE_RE.match(doc)
    return m.group(1).strip() if m else ""


@dataclass
class RetrievalResult:
    """A retrieved passage with its component scores."""

    text: str
    hybrid_score: float
    dense_score: float
    bm25_score: float
    idx: int

    @property
    def title(self) -> str:
        return extract_title(self.text)

    # Dict-style access keeps older code (r["text"]) working.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


class HybridIndex:
    """Immutable BM25 + dense index over one document set."""

    def __init__(self, documents: Sequence[str], dense: DenseIndex, store: VectorStore) -> None:
        self.documents = list(documents)
        self._dense = dense
        self._store = store
        tokenized = [tokenize(d) or ["_"] for d in self.documents]
        self._bm25 = BM25Okapi(tokenized) if self.documents else None

    def __len__(self) -> int:
        return len(self.documents)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Args:
            query: search text.
            top_k: number of results (defaults to config.TOP_K_RETRIEVAL).
            method: "linear" | "rrf" | "bm25" | "dense" (defaults to config).
            exclude: passages to skip (e.g. ones already in the context).
        """
        n = len(self.documents)
        if n == 0:
            return []
        top_k = top_k or config.TOP_K_RETRIEVAL
        method = (method or config.RETRIEVAL_FUSION_METHOD).lower()

        bm25_raw = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        bm25 = _minmax(bm25_raw)

        dense_raw = np.zeros(n, dtype=float)
        if method != "bm25":
            for hit in self._dense.search(self._store.embed_query(query), top_k=n):
                dense_raw[hit["idx"]] = hit["score"]
        dense = _minmax(dense_raw)

        if method == "bm25":
            fused = bm25_raw
        elif method == "dense":
            fused = dense_raw
        elif method == "rrf":
            fused = np.zeros(n, dtype=float)
            for scores in (bm25_raw, dense_raw):
                for rank, idx in enumerate(np.argsort(-scores, kind="stable")):
                    fused[idx] += 1.0 / (config.RRF_K + rank + 1)
        else:
            fused = config.BM25_WEIGHT * bm25 + config.DENSE_WEIGHT * dense

        skip = set(exclude or [])
        results: List[RetrievalResult] = []
        for idx in np.argsort(-fused, kind="stable"):
            text = self.documents[idx]
            if text in skip:
                continue
            results.append(RetrievalResult(
                text=text,
                hybrid_score=float(fused[idx]),
                dense_score=float(dense_raw[idx]),
                bm25_score=float(bm25[idx]),
                idx=int(idx),
            ))
            if len(results) >= top_k:
                break
        return results

    def search_texts(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> List[str]:
        return [r.text for r in self.search(query, top_k=top_k, **kwargs)]


class HybridRetriever:
    """Factory for HybridIndex objects; holds the (shared) embedding model."""

    def __init__(self, vector_store: Optional[VectorStore] = None, encoder: Any = None) -> None:
        self.vector_store = vector_store or VectorStore(encoder=encoder)
        self._last: Optional[HybridIndex] = None  # legacy stateful API only

    def build_index(self, documents: Sequence[str]) -> HybridIndex:
        docs = [d for d in documents if d and d.strip()]
        return HybridIndex(docs, self.vector_store.create_index(docs), self.vector_store)

    # Legacy stateful API (not thread-safe) — kept so older scripts still work.
    def index_documents(self, documents: Sequence[str]) -> None:
        self._last = self.build_index(documents)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        if self._last is None:
            logger.warning("No documents indexed. Returning empty results.")
            return []
        return self._last.search(query, top_k=top_k)
