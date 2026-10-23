import pytest

from agent.self_monitor import MonitorSettings, SelfMonitor, expects_yes_no
from tests.fakes import FakeChatClient, make_llm

Q = "What is the nationality of the director of Inception?"


def _monitor(nli, mode="tiered", client=None):
    client = client or FakeChatClient()
    return SelfMonitor(make_llm(client), nli=nli, settings=MonitorSettings.from_mode(mode)), client


def _judge_calls(client):
    return sum("VERDICT" in p for p in client.prompts)


@pytest.mark.parametrize("answer,ftype", [("", "empty"), ("I don't know", "refusal"),
                                          ("The context does not say", "refusal")])
def test_tier1_catches_empty_and_refusals(nli, docs, answer, ftype):
    m, client = _monitor(nli)
    r = m.evaluate(Q, answer, docs)
    assert r.has_failure and r.failure_type == ftype and r.monitor_tier_used == 1
    assert client.prompts == []  # zero LLM calls


def test_tier1_answer_type_mismatch(nli, docs):
    m, _ = _monitor(nli)
    assert m.evaluate("Were both films directed by Nolan?", "Christopher Nolan", docs).failure_type == "type_mismatch"
    assert m.evaluate("Is Paris or Berlin the capital of France?", "yes", docs).failure_type == "type_mismatch"
    assert expects_yes_no("Who is older?") is None


def test_tier2_passes_grounded_answer_without_llm(nli, docs):
    m, client = _monitor(nli)
    r = m.evaluate(Q, "British", docs)
    assert not r.has_failure and r.monitor_tier_used == 2 and r.confidence_score > 0.9
    assert _judge_calls(client) == 0


def test_tier2_fails_unsupported_answer_without_llm(nli, docs):
    m, client = _monitor(nli)
    r = m.evaluate(Q, "Japanese", docs)
    assert r.has_failure and r.failure_type == "ungrounded" and r.monitor_tier_used == 2
    assert _judge_calls(client) == 0


def test_tier3_only_runs_in_uncertain_band(nli, docs):
    m, client = _monitor(nli)
    # "British filmmaker from Tokyo" partly overlaps -> FakeNLI p=0.4 (uncertain band) -> judge
    r = m.evaluate(Q, "British filmmaker from Tokyo", docs)
    assert r.monitor_tier_used == 3 and _judge_calls(client) == 1
    assert r.has_failure and r.failure_type == "hallucinated"  # 'tokyo' not in context


def test_llm_only_mode_always_calls_judge(nli, docs):
    m, client = _monitor(nli, mode="llm")
    r = m.evaluate(Q, "British", docs)
    assert not r.has_failure and r.monitor_tier_used == 3 and _judge_calls(client) == 1


def test_nli_only_mode_resolves_uncertain_band_without_llm(nli, docs):
    m, client = _monitor(nli, mode="heuristic_nli")
    low = m.evaluate(Q, "British filmmaker from Tokyo", docs)        # p=0.40 < midpoint 0.45 -> fail
    high = m.evaluate(Q, "British filmmaker Christopher Nolan from Tokyo", docs)  # p=0.55 -> pass
    assert low.has_failure and not high.has_failure and _judge_calls(client) == 0


def test_long_answer_checked_sentence_by_sentence(nli, docs):
    m, _ = _monitor(nli, mode="heuristic_nli")
    m.settings.short_answers = False
    answer = ("Inception was directed by Christopher Nolan in 2010. "
              "Nolan won seven Academy Awards for directing Inception.")
    r = m.evaluate("Who directed Inception and what awards did he win?", answer, docs)
    assert r.has_failure and any("Academy" in s for s in r.ungrounded_sentences)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        MonitorSettings.from_mode("magic")
