import pytest
from fastapi.testclient import TestClient

from api import server
from pipeline import Components, SelfHealingRAGPipeline
from tests.fakes import FakeChatClient, make_llm


@pytest.fixture
def client(retriever, nli):
    fake = FakeChatClient(answers={"nationality of the director": ["German", "British"]})
    pipe = SelfHealingRAGPipeline(components=Components(llm=make_llm(fake), retriever=retriever, nli=nli))
    server.app.dependency_overrides[server.get_pipeline] = lambda: pipe
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "model" in body


def test_ask_heals_and_reports_rounds(client, docs):
    r = client.post("/v1/ask", json={"question": "What is the nationality of the director of Inception?",
                                      "documents": docs, "question_type": "direct"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "British" and body["initial_answer"] == "German" and body["healed"]
    assert body["verified"] is True
    assert [x["has_failure"] for x in body["rounds"]] == [True, False]
    assert body["cost"]["n_llm_calls"] >= 2 and body["trace"]


def test_verify_endpoint(client, docs):
    ok = client.post("/v1/verify", json={"question": "What is the capital of France?", "answer": "Paris",
                                          "documents": docs, "mode": "heuristic_nli"}).json()
    bad = client.post("/v1/verify", json={"question": "What is the capital of France?", "answer": "Lyon",
                                           "documents": docs, "mode": "heuristic_nli"}).json()
    assert ok["grounded"] and not bad["grounded"] and bad["failure_type"] == "ungrounded"


def test_validation_errors(client):
    assert client.post("/v1/ask", json={"question": "hi", "documents": []}).status_code == 422
    assert client.post("/v1/ask", json={"question": "What?", "documents": ["x" * 30_000]}).status_code == 422
