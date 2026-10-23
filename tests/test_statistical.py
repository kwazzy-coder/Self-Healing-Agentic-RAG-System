import pytest

from evaluation.statistical import (
    aggregate_across_seeds,
    bootstrap_confidence_interval,
    mcnemar_test,
    paired_bootstrap_test,
)


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_confidence_interval([0, 1] * 50)
    assert lo < mean < hi
    assert mean == pytest.approx(0.5)


def test_mcnemar_exact_and_asymptotic():
    a = [True] * 10 + [False] * 2
    b = [False] * 10 + [True] * 2
    small = mcnemar_test(a, b)
    assert small["a_only"] == 10 and small["b_only"] == 2
    assert small["p_value"] < 0.05
    big = mcnemar_test([True] * 40 + [False] * 10, [False] * 40 + [True] * 10)
    assert big["significant"]
    assert mcnemar_test([True], [True])["p_value"] == 1.0


def test_paired_bootstrap_detects_real_difference_only():
    better = [1.0] * 80 + [0.0] * 20
    worse = [0.0] * 80 + [0.0] * 20
    assert paired_bootstrap_test(better, worse)["significant"]
    assert not paired_bootstrap_test(better, better)["significant"]


def test_aggregate_across_seeds_ignores_non_numeric():
    agg = aggregate_across_seeds([{"em": 0.5, "name": "x"}, {"em": 0.7, "name": "y"}])
    assert agg["em"]["mean"] == pytest.approx(0.6)
    assert "name" not in agg
