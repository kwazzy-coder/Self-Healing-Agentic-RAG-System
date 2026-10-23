"""
retrieval/chunking.py — Split long documents into retrieval-sized chunks.

Embedding models such as all-MiniLM-L6-v2 only read the first ~256 tokens of
a passage, so long user documents are split on sentence boundaries with a
one-sentence overlap. Each chunk keeps the document's "Title:" line so the
generator (and the supporting-fact metrics) still know where it came from.
"""
from __future__ import annotations

import re
from typing import List, Sequence

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def chunk_document(doc: str, max_chars: int = 1000, overlap_sentences: int = 1) -> List[str]:
    """Split one document; short documents are returned unchanged."""
    if len(doc) <= max_chars:
        return [doc]
    title = ""
    body = doc
    if doc.startswith("Title:") and "\n" in doc:
        title, body = doc.split("\n", 1)
        title += "\n"
    sentences = split_sentences(body)
    chunks: List[str] = []
    current: List[str] = []
    for sent in sentences:
        if current and len(" ".join(current + [sent])) + len(title) > max_chars:
            chunks.append(title + " ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sent)
    if current:
        chunks.append(title + " ".join(current))
    return chunks or [doc]


def chunk_documents(docs: Sequence[str], max_chars: int = 1000, overlap_sentences: int = 1) -> List[str]:
    out: List[str] = []
    for d in docs:
        out.extend(chunk_document(d, max_chars=max_chars, overlap_sentences=overlap_sentences))
    return out
