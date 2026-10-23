from evaluation.metrics import (
    all_gold_at_k,
    binary_classification_metrics,
    contains_match,
    exact_match,
    normalize_answer,
    retrieval_mrr,
    retrieval_recall_at_k,
    roc_auc,
    supporting_doc_f1,
    token_f1,
)


def test_normalize_matches_official_squad_rules():
    assert normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert normalize_answer("An apple") == "apple"


def test_exact_match_and_f1():
    assert exact_match("Paris.", "paris") == 1.0
    assert exact_match("Paris, France", "Paris") == 0.0
    assert token_f1("Paris France", "Paris") == 2 * (0.5 * 1.0) / 1.5
    assert token_f1("", "") == 1.0


def test_yes_no_special_case_in_f1():
    # Official HotpotQA: yes/no answers get no partial credit
    assert token_f1("yes it is", "yes") == 0.0
    assert token_f1("no", "no") == 1.0


def test_contains_match():
    assert contains_match("the city of Paris", "Paris") == 1.0
    assert contains_match("a very long answer that mentions Paris somewhere in it", "Paris") == 0.0


def test_retrieval_metrics_use_exact_titles():
    gold = ["Paris", "France"]
    titles = ["Paris Hilton", "France", "Paris"]
    assert retrieval_recall_at_k(titles, gold, k=2) == 0.5      # "Paris Hilton" is not "Paris"
    assert retrieval_recall_at_k(titles, gold, k=3) == 1.0
    assert all_gold_at_k(titles, gold, 2) == 0.0
    assert retrieval_mrr(titles, gold) == 0.5
    assert 0 < supporting_doc_f1(titles, gold) < 1


def test_binary_metrics_and_auroc():
    y = [True, True, False, False]
    m = binary_classification_metrics(y, [True, False, False, True])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.5
    assert roc_auc(y, [0.9, 0.8, 0.1, 0.2]) == 1.0
    assert roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert roc_auc([True], [0.3]) is None
