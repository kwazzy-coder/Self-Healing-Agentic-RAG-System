from agent.corrective_actions import CorrectiveActions
from agent.task_planner import TaskPlanner, substitute_previous_answer
from tests.fakes import FakeChatClient, make_llm


def test_strategy_detection():
    d = TaskPlanner.detect_strategy
    assert d("x", "bridge") == "sequential" and d("x", "comparison") == "parallel"
    assert d("Who is older, Alice or Bob?") == "parallel"
    assert d("What is the nationality of the director of Inception?") == "sequential"
    assert d("What is the capital of France?") == "direct"


def test_plan_uses_llm_decomposition():
    client = FakeChatClient(decompositions={"Inception": ["Who directed Inception?", "What nationality is #1?"]})
    plan = TaskPlanner(make_llm(client)).plan("What nationality is the director of Inception?", "bridge")
    assert plan.strategy == "sequential" and plan.sub_questions[1] == "What nationality is #1?"
    assert plan["strategy"] == "sequential"  # dict-style access still works


def test_substitute_previous_answer():
    assert substitute_previous_answer("What nationality is #1?", "Christopher Nolan") == \
        "What nationality is Christopher Nolan?"
    assert substitute_previous_answer("Where was [answer] born?", "Ada") == "Where was Ada born?"


def test_strategy_escalates_instead_of_repeating():
    c = CorrectiveActions(llm=None)
    assert c.choose_strategy("ungrounded") == "re_retrieve"
    c._tried.append("re_retrieve")
    assert c.choose_strategy("ungrounded") == "decompose"
    assert c.choose_strategy("refusal") == "reformulate"


def test_targeted_reretrieval_brings_new_evidence(retriever, docs):
    client = FakeChatClient(answers={"SEARCH::": "Christopher Nolan nationality", "nationality": "British"})
    corrector = CorrectiveActions(make_llm(client), top_k=2, max_docs=4)
    index = retriever.build_index(docs)
    current = [d for d in docs if d.startswith("Title: Inception")]
    res = corrector.apply("re_retrieve", "What is the nationality of the director of Inception?", "German",
                          current, failure_type="ungrounded", index=index)
    assert res.strategy == "re_retrieve"
    assert any(d.startswith("Title: Christopher Nolan") for d in res.new_docs)
    assert current[0] in res.new_docs  # existing evidence kept
    assert res.new_answer == "British"


def test_decompose_substitutes_bridge_entity(retriever, docs):
    client = FakeChatClient(
        answers={"Who directed Inception": "Christopher Nolan", "What nationality is Christopher Nolan": "British",
                 "nationality of the director": "British"},
        decompositions={"director of Inception": ["Who directed Inception?", "What nationality is #1?"]},
    )
    corrector = CorrectiveActions(make_llm(client), top_k=2, max_docs=5)
    res = corrector.apply("decompose", "What is the nationality of the director of Inception?", "?", [],
                          failure_type="type_mismatch", index=retriever.build_index(docs))
    assert res.strategy == "decompose"
    assert res.sub_qas[1] == ("What nationality is Christopher Nolan?", "British")
    assert res.new_answer == "British"
