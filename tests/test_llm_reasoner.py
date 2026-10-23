from types import SimpleNamespace

import pytest

import config
from agent import llm_reasoner
from agent.llm_reasoner import LLMError, LLMReasoner, parse_numbered_lines, parse_verdict
from tests.fakes import FakeChatClient, make_llm


@pytest.fixture
def llm():
    return make_llm(FakeChatClient())


@pytest.mark.parametrize("raw,expected", [
    ("Reasoning: blah.\nAnswer: Paris", "Paris"),
    ("Reasoning: x\nAnswer: Yes, both are directors.", "yes"),
    ("Answer: no.", "no"),
    ("Answer: No Doubt", "No Doubt"),                     # a band name, not a "no"
    ("Answer: The answer is Christopher Nolan.", "Christopher Nolan"),
    ("Answer: Paris is the capital of France", "Paris"),
    ("Answer: **Canadian**", "Canadian"),
    ("Answer: Berlin (Germany)", "Berlin"),
    ("just one line", "just one line"),
])
def test_extract_final_answer(llm, raw, expected):
    assert llm.extract_final_answer(raw) == expected


def test_parse_numbered_lines():
    assert parse_numbered_lines("1. Who directed X?\n2) What nationality is #1?\nnoise") == [
        "Who directed X?", "What nationality is #1?"]
    assert parse_numbered_lines("**1.** Who is A?\n**2.** Who is B?") == ["Who is A?", "Who is B?"]


@pytest.mark.parametrize("raw,verdict", [
    ("VERDICT: SUPPORTED - found in [1]", True),
    ("VERDICT: UNSUPPORTED - not mentioned", False),
    ("NOT SUPPORTED", False),
    ("Yes.", True),
    ("No", False),
    ("hmm", None),
])
def test_parse_verdict(raw, verdict):
    assert parse_verdict(raw)[0] is verdict


class _Err(Exception):
    def __init__(self, status, msg="boom"):
        super().__init__(msg)
        self.status_code = status


def _client_raising(errors, reply="Answer: ok"):
    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_retries_rate_limits_then_succeeds():
    client, calls = _client_raising([_Err(429), _Err(503)])
    llm = LLMReasoner(backend="groq", model="m", client=client)
    assert llm.generate("hi") == "Answer: ok"
    assert calls["n"] == 3 and llm.usage.retries == 2 and llm.usage.calls == 1


def test_non_retryable_error_raises_llmerror():
    client, calls = _client_raising([_Err(401, "invalid api key")])
    llm = LLMReasoner(backend="groq", model="m", client=client)
    with pytest.raises(LLMError):
        llm.generate("hi")
    assert calls["n"] == 1


def test_drops_temperature_when_model_rejects_it():
    seen = []

    def create(**kwargs):
        seen.append("temperature" in kwargs)
        if "temperature" in kwargs:
            raise _Err(400, "Unsupported value: 'temperature' does not support 0")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)

    llm = LLMReasoner(backend="openai", model="m",
                      client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert llm.generate("hi") == "ok"
    assert seen == [True, False]


def test_reasoning_budget_and_token_param(monkeypatch):
    monkeypatch.setattr(config, "LLM_REASONING_EFFORT", "low")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert LLMReasoner(backend="groq", model="m", client=client).generate("x", max_tokens=5) == ""
    assert captured["extra_body"]["max_completion_tokens"] >= config.MIN_OUTPUT_TOKENS  # thinking headroom
    assert captured["extra_body"]["reasoning_effort"] == "low"
    LLMReasoner(backend="gemini", model="m", client=client).generate("x")
    assert "max_tokens" in captured


def test_provider_fallback_without_keys(monkeypatch):
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(config, key, "")
    monkeypatch.setattr(llm_reasoner, "_ollama_available", lambda: False)
    assert llm_reasoner.resolve_provider("gemini") == "huggingface"
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    assert llm_reasoner.resolve_provider("gemini") == "groq"
    assert llm_reasoner.resolve_provider("groq") == "groq"


def test_default_models_are_current():
    # Both original defaults were shut down in 2026; guard against regressions.
    assert config.PROVIDER_DEFAULT_MODELS["gemini"] != "gemini-2.0-flash-lite"
    assert config.PROVIDER_DEFAULT_MODELS["groq"] != "llama-3.1-8b-instant"
