from retrieval.chunking import chunk_document
from retrieval.retriever import RetrievalResult, extract_title


def test_hybrid_ranks_relevant_doc_first(retriever, docs):
    index = retriever.build_index(docs)
    hits = index.search("Who directed the film Inception?", top_k=2)
    assert hits[0].title == "Inception"
    assert isinstance(hits[0], RetrievalResult) and hits[0]["text"] == hits[0].text  # dict-style access


def test_all_fusion_methods_and_exclude(retriever, docs):
    index = retriever.build_index(docs)
    for method in ("bm25", "dense", "linear", "rrf"):
        titles = [r.title for r in index.search("capital of France", top_k=3, method=method)]
        assert "France" in titles, method
    first = index.search_texts("capital of France", top_k=1)
    again = index.search_texts("capital of France", top_k=1, exclude=first)
    assert again and again[0] != first[0]


def test_index_is_immutable_and_independent(retriever, docs):
    a = retriever.build_index(docs[:2])
    b = retriever.build_index(docs[2:])
    assert len(a) == 2 and len(b) == len(docs) - 2
    assert a.search("Eiffel", top_k=1)[0].title == "Eiffel Tower"


def test_empty_index(retriever):
    assert retriever.build_index([]).search("anything") == []


def test_embedding_cache_avoids_recomputation(retriever, encoder, docs):
    retriever.build_index(docs)
    calls = encoder.calls
    retriever.build_index(docs)
    assert encoder.calls == calls  # passages served from cache


def test_chunking_keeps_title():
    doc = "Title: Long\n" + " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_document(doc, max_chars=300)
    assert len(chunks) > 3
    assert all(c.startswith("Title: Long\n") and len(c) <= 330 for c in chunks)
    assert extract_title(chunks[-1]) == "Long"
    assert chunk_document("Title: Short\nTiny.") == ["Title: Short\nTiny."]
