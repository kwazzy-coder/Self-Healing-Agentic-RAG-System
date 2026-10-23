"""
Exercises the REAL openai SDK code path against a local fake server that speaks
the OpenAI-compatible Chat Completions protocol (as Gemini/Groq/Ollama do).
Checks request shape, retry on 429 with Retry-After, and error classification.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import config
from agent.llm_reasoner import LLMError, LLMReasoner


class _Handler(BaseHTTPRequestHandler):
    script: list = []
    requests: list = []

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        status, payload = type(self).script.pop(0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if status == 429:
            self.send_header("Retry-After", "0")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())


def _ok(text):
    return 200, {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": text}}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}


@pytest.fixture
def server(monkeypatch):
    pytest.importorskip("openai")
    _Handler.script, _Handler.requests = [], []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    monkeypatch.setitem(config.PROVIDER_BASE_URLS, "groq", url)
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", "low")
    yield _Handler
    httpd.shutdown()


def test_real_sdk_request_and_retry(server):
    server.script = [(429, {"error": {"message": "rate limited"}}), _ok("Reasoning: x\nAnswer: Paris")]
    llm = LLMReasoner(backend="groq", model="openai/gpt-oss-20b")
    assert llm.answer_with_context("What is the capital of France?", ["Title: France\nCapital: Paris."]) == "Paris"
    assert len(server.requests) == 2 and llm.usage.retries == 1
    req = server.requests[-1]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer gsk_test"
    body = req["body"]
    assert body["model"] == "openai/gpt-oss-20b" and body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] >= config.MIN_OUTPUT_TOKENS and body["temperature"] == 0.0
    assert "[1] Title: France" in body["messages"][0]["content"]
    assert llm.usage.prompt_tokens == 11 and llm.usage.completion_tokens == 3


def test_real_sdk_auth_error_is_not_retried(server):
    server.script = [(401, {"error": {"message": "Invalid API Key"}})]
    llm = LLMReasoner(backend="groq", model="m")
    with pytest.raises(LLMError, match="Invalid API Key"):
        llm.generate("hi")
    assert len(server.requests) == 1
