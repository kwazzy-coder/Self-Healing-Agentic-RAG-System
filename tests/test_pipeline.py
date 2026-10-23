import threading

from pipeline import PipelineConfig
from tests.fakes import FakeChatClient

Q = "What is the nationality of the director of Inception?"


def test_healing_fixes_a_wrong_initial_answer(make_pipeline, docs):
    client = FakeChatClient(answers={"nationality of the director": ["German", "British"]})
    out = make_pipeline(client, enable_planning=False).run(Q, docs)
    assert out.initial_answer == "German"
    assert out.final_answer == "British"
    assert out.failure_detected and out.correction_attempted and out.selected_round == 1
    assert [r.has_failure for r in out.rounds] == [True, False]   # the correction was verified too
    assert out.cost_data["n_llm_calls"] >= 3 and out.cost_data["correction_rounds"] == 1


def test_correct_answer_passes_without_correction(make_pipeline, docs):
    client = FakeChatClient(answers={"capital of France": "Paris"})
    out = make_pipeline(client, enable_planning=False).run("What is the capital of France?", docs)
    assert out.final_answer == "Paris" and not out.failure_detected and out.n_correction_rounds == 0
    assert out.cost_data["n_llm_calls"] == 1                        # Tier 2 decided, no judge call


def test_worse_correction_never_replaces_better_answer(make_pipeline, docs):
    # Initial answer is right but phrased so NLI is uncertain and the judge rejects it;
    # every correction is worse (refusal). The pipeline must keep the best candidate.
    client = FakeChatClient(answers={"nationality of the director": ["British filmmaker Nolan from Tokyo",
                                                                     "I don't know"]})
    out = make_pipeline(client, enable_planning=False).run(Q, docs)
    assert out.failure_detected
    assert out.final_answer == "British filmmaker Nolan from Tokyo"
    assert out.selected_round == 0 and out.verified is False   # flagged to the caller as unverified
    assert len(out.rounds) == 3                                      # 2 corrections, all checked


def test_vanilla_baseline_skips_monitor(make_pipeline, docs):
    client = FakeChatClient(answers={"capital of France": "Paris"})
    out = make_pipeline(client).run_baseline("What is the capital of France?", docs)
    assert out.final_answer == "Paris" and out.system == "vanilla_rag"
    assert out.rounds[0].has_failure is None and out.cost_data["n_llm_calls"] == 1


def test_always_correct_and_closed_book(make_pipeline, docs):
    client = FakeChatClient(answers={"capital of France": "Paris"})
    out = make_pipeline(client, always_correct=True, enable_planning=False).run("What is the capital of France?", docs)
    assert out.n_correction_rounds == 1 and out.selected_round == 1
    cb = make_pipeline(client, use_retrieval=False).run("What is the capital of France?", docs)
    assert cb.retrieved_docs == [] and cb.final_answer == "Paris"


def test_per_call_config_override(make_pipeline, docs):
    client = FakeChatClient(answers={"nationality of the director": ["German", "British"]})
    p = make_pipeline(client, enable_planning=False)
    out = p.run(Q, docs, pipeline_config=PipelineConfig(enable_planning=False, max_correction_rounds=0))
    assert out.final_answer == "German" and out.n_correction_rounds == 0


def test_concurrent_requests_do_not_mix_state(make_pipeline, docs):
    client = FakeChatClient(answers={"capital of France": "Paris", "capital of Germany": "Berlin"})
    p = make_pipeline(client, enable_planning=False)
    results = {}

    def work(i):
        q = "What is the capital of France?" if i % 2 else "What is the capital of Germany?"
        results[i] = p.run(q, docs)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, out in results.items():
        assert out.final_answer == ("Paris" if i % 2 else "Berlin")
        assert out.cost_data["n_llm_calls"] == 1   # per-query cost not polluted by other threads


def test_inconsistent_multihop_synthesis_is_repaired_without_llm(make_pipeline, docs):
    # hop answers are right, but the synthesis step says "German"
    client = FakeChatClient(
        answers={"Who directed Inception": "Christopher Nolan", "What nationality is Christopher Nolan": "British",
                 "nationality of the director": "German"},
        decompositions={"director of Inception": ["Who directed Inception?", "What nationality is #1?"]},
    )
    out = make_pipeline(client).run(Q, docs, question_type="bridge")
    assert out.rounds[0].failure_type == "inconsistent"
    assert out.rounds[1].produced_by == "adopt_last_hop" and out.final_answer == "British"
    assert out.cost_data["n_llm_calls"] == 4   # decompose + 2 hops + synthesis; the repair itself is free


def test_correction_stops_when_no_strategy_can_help(make_pipeline, docs):
    small = docs[:2]                       # everything fits in context -> retrieval strategies are futile
    client = FakeChatClient(answers={"capital of Spain": "I don't know"})
    out = make_pipeline(client, enable_planning=False, top_k=5).run("What is the capital of Spain?", small)
    produced = [r.produced_by for r in out.rounds]
    assert "re_retrieve" not in produced and "reformulate" not in produced
    assert produced[:2] == ["initial", "decompose"]
