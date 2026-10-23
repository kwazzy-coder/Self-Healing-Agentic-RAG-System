from types import SimpleNamespace

import numpy as np

from agent.nli import NLIScorer, answer_to_hypotheses, build_premises


def test_short_answers_become_claims_and_long_ones_split():
    hyp = answer_to_hypotheses("Who directed Inception?", "Christopher Nolan")
    assert hyp == ['The answer to the question "Who directed Inception?" is Christopher Nolan.']
    long = answer_to_hypotheses("q?", "Paris is the capital of France. It lies on the Seine river. "
                                      "The city has two million residents in total.")
    assert len(long) == 3


def test_premises_include_joint_evidence():
    prem = build_premises(["Title: A\nA directed B.", "Title: C\nA was born in Canada."], max_chars=500)
    assert len(prem) == 3 and "directed" in prem[-1] and "Canada" in prem[-1]


def test_label_order_is_read_from_model_config():
    class Model:  # entailment is label 0 here, unlike the default layout
        config = SimpleNamespace(id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})

        def predict(self, pairs, **_):
            return np.array([[0.9, 0.05, 0.05]] * len(pairs))

    score = NLIScorer(model=Model()).score(["p1", "p2"], ["h"])[0]
    assert score.entailment == 0.9 and score.contradiction == 0.05


def test_empty_inputs():
    assert NLIScorer(model=object()).score([], ["h"])[0].entailment == 0.0
