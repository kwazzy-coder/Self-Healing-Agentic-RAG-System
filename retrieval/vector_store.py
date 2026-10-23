"""
retrieval/vector_store.py — Dense vector index (FAISS, with a NumPy fallback).

* The sentence-transformers model is loaded once per process and shared.
* Embeddings are cached by text hash, so running the baseline and the
  self-healing system on the same sample embeds each passage only once.
* Any object with ``encode(texts, normalize_embeddings=True, ...)`` can be
  injected as the encoder (tests use a tiny hashing encoder, no download).
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import config

logger = logging.getLogger(__name__)

try:  # FAISS is optional: NumPy is just as fast for the small per-query indexes used here.
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:  # pragma: no cover - depends on platform
    faiss = None
    HAS_FAISS = False

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def get_encoder(model_name: Optional[str] = None) -> Any:
    """Load (once) and return a SentenceTransformer encoder."""
    name = model_name or config.EMBEDDING_MODEL
    with _MODEL_LOCK:
        if name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", name)
            _MODEL_CACHE[name] = SentenceTransformer(name)
        return _MODEL_CACHE[name]


class EmbeddingCache:
    """Small thread-safe LRU cache of text -> normalised embedding."""

    def __init__(self, max_items: int = 50_000) -> None:
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, encoder: Any, texts: Sequence[str]) -> np.ndarray:
        keys = [self.key(t) for t in texts]
        missing_idx: List[int] = []
        with self._lock:
            for i, k in enumerate(keys):
                if k in self._data:
                    self._data.move_to_end(k)
                    self.hits += 1
                else:
                    missing_idx.append(i)
                    self.misses += 1
        if missing_idx:
            new_vecs = encoder.encode(
                [texts[i] for i in missing_idx], normalize_embeddings=True, show_progress_bar=False
            )
            new_vecs = np.asarray(new_vecs, dtype="float32")
            with self._lock:
                for i, vec in zip(missing_idx, new_vecs, strict=False):
                    self._data[keys[i]] = vec
                while len(self._data) > self.max_items:
                    self._data.popitem(last=False)
        with self._lock:
            return np.stack([self._data[k] for k in keys]).astype("float32")


_SHARED_CACHE = EmbeddingCache()


class DenseIndex:
    """Inner-product index over L2-normalised embeddings (= cosine similarity)."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        self.size = embeddings.shape[0]
        self._faiss = None
        if HAS_FAISS and self.size:
            self._faiss = faiss.IndexFlatIP(embeddings.shape[1])
            self._faiss.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, float]]:
        if self.size == 0:
            return []
        k = min(top_k, self.size)
        q = query_vec.reshape(1, -1).astype("float32")
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)
            pairs = zip(scores[0], idxs[0], strict=False)
        else:
            sims = self.embeddings @ q[0]
            order = np.argsort(-sims)[:k]
            pairs = ((sims[i], i) for i in order)
        return [{"idx": int(i), "score": float(s)} for s, i in pairs if i >= 0]


class VectorStore:
    """Builds dense indexes and embeds queries with a shared encoder."""

    def __init__(self, model_name: Optional[str] = None, encoder: Any = None,
                 cache: Optional[EmbeddingCache] = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._encoder = encoder
        self.cache = cache or _SHARED_CACHE
        # Legacy stateful API (build_index + search) kept for older scripts.
        self._last_index: Optional[DenseIndex] = None
        self.documents: List[str] = []

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = get_encoder(self.model_name)
        return self._encoder

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.cache.encode(self.encoder, list(texts))

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype="float32")[0]

    def create_index(self, documents: Sequence[str]) -> DenseIndex:
        if not documents:
            return DenseIndex(np.zeros((0, 1), dtype="float32"))
        return DenseIndex(self.embed(documents))

    # Legacy API ---------------------------------------------------------------
    def build_index(self, documents: List[str]) -> None:
        self.documents = list(documents)
        self._last_index = self.create_index(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self._last_index is None:
            return []
        hits = self._last_index.search(self.embed_query(query), top_k)
        return [{"text": self.documents[h["idx"]], **h} for h in hits]
