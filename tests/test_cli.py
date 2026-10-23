"""Smoke tests for the command-line entry points, with every network dependency faked."""
import pytest

import config
import demo
import pipeline
import run_evaluation
from agent import llm_reasoner, nli
from retrieval.retriever import HybridRetriever
from retrieval.vector_store import EmbeddingCache, VectorStore
from tests.fakes import FakeChatClient, FakeNLIModel, HashEncoder, make_llm
from tests.test_evaluator import SAMPLES


@pytest.fixture
def offline(monkeypatch, tmp_path):
    client = FakeChatClient(answers={"nationality": "British", "capital of France": "Paris",
                                     "Eiffel": "Paris", "telephone": "I don't know", "COVID": "SARS-CoV-2",
                                     "Statue of Liberty": "after"})
    monkeypatch.setattr(llm_reasoner, "LLMReasoner", lambda *a, **k: make_llm(client))
    monkeypatch.setattr(pipeline, "LLMReasoner", lambda *a, **k: make_llm(client))
    monkeypatch.setattr(pipeline, "HybridRetriever",
                        lambda: HybridRetriever(VectorStore(encoder=HashEncoder(), cache=EmbeddingCache())))
    real_scorer = nli.NLIScorer
    fake_nli = lambda *a, **k: real_scorer(model=FakeNLIModel())  # noqa: E731
    monkeypatch.setattr(pipeline, "NLIScorer", fake_nli)
    monkeypatch.setattr(nli, "NLIScorer", fake_nli)
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    return tmp_path


def test_run_evaluation_qa(offline, monkeypatch, capsys):
    monkeypatch.setattr(run_evaluation, "load_samples", lambda args: SAMPLES)
    assert run_evaluation.main(["--dataset", "hotpotqa", "--n_samples", "2", "--compare_baseline",
                                "--systems", "self_healing,crag"]) == 0
    out = capsys.readouterr().out
    assert "vanilla_rag" in out and "self_healing" in out and "crag" in out
    assert (offline / "hotpotqa_qa_n2_s42" / "summary.json").exists()


def test_run_evaluation_detector(offline, monkeypatch, capsys):
    rows = [{"id": "a", "question": "What is the capital of France?", "gold_response": "Paris.",
             "documents": [s for s in demo.DEMO_SAMPLES[0]["documents"]], "adherence_score": True}]
    monkeypatch.setattr(run_evaluation, "load_samples", lambda args: rows)
    assert run_evaluation.main(["--dataset", "ragbench", "--mode", "detector", "--monitors",
                                "heuristic_nli,tiered"]) == 0
    assert "AUROC" in capsys.readouterr().out


def test_detector_mode_requires_ragbench(offline, monkeypatch):
    monkeypatch.setattr(run_evaluation, "load_samples", lambda args: SAMPLES)
    assert run_evaluation.main(["--dataset", "hotpotqa", "--mode", "detector"]) == 1


def test_demo_runs(offline, capsys):
    demo.run_demo()
    assert "EM" in capsys.readouterr().out
