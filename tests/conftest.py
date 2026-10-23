"""Shared pytest fixtures. Nothing here touches the network or downloads models."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402
from agent.nli import NLIScorer  # noqa: E402
from pipeline import Components, SelfHealingRAGPipeline  # noqa: E402
from retrieval.retriever import HybridRetriever  # noqa: E402
from retrieval.vector_store import EmbeddingCache, VectorStore  # noqa: E402
from tests.fakes import FakeChatClient, FakeNLIModel, HashEncoder, make_llm  # noqa: E402

DOCS = [
    "Title: Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
    "It was built in 1889.",
    "Title: France\nFrance is a country in Western Europe. Its capital and largest city is Paris.",
    "Title: Inception\nInception is a 2010 science fiction film written and directed by Christopher Nolan.",
    "Title: Christopher Nolan\nChristopher Nolan is a British-American filmmaker born in London, England.",
    "Title: Germany\nGermany is a country in Central Europe. Its capital is Berlin.",
    "Title: Statue of Liberty\nThe Statue of Liberty was dedicated on October 28, 1886.",
]


@pytest.fixture(autouse=True)
def _fast_config(monkeypatch):
    """No sleeping, no retries waiting, deterministic settings for every test."""
    monkeypatch.setattr(config, "MIN_DELAY_BETWEEN_CALLS", 0.0)
    monkeypatch.setattr(config, "RETRY_BASE_WAIT_S", 0.0)
    monkeypatch.setattr(config, "RETRY_MAX_WAIT_S", 0.0)
    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", None)
    monkeypatch.setattr(config, "TOP_K_RETRIEVAL", 3)
    monkeypatch.setattr(config, "MAX_DOCS_PER_QUERY", 5)
    monkeypatch.setattr(config, "MAX_CORRECTION_ATTEMPTS", 2)
    yield


@pytest.fixture
def docs():
    return list(DOCS)


@pytest.fixture
def encoder():
    return HashEncoder()


@pytest.fixture
def retriever(encoder):
    return HybridRetriever(vector_store=VectorStore(encoder=encoder, cache=EmbeddingCache()))


@pytest.fixture
def nli():
    return NLIScorer(model=FakeNLIModel())


@pytest.fixture
def make_pipeline(retriever, nli):
    def _make(client: FakeChatClient, **cfg):
        from pipeline import PipelineConfig

        llm = make_llm(client)
        return SelfHealingRAGPipeline(
            pipeline_config=PipelineConfig(**cfg) if cfg else None,
            components=Components(llm=llm, retriever=retriever, nli=nli),
        )

    return _make
