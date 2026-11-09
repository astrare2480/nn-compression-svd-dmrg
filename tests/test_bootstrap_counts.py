"""counts（各単位が選ばれた回数の行列）で持つブートストラップ。"""

from __future__ import annotations

import numpy as np
import pytest

from nn_compression.metrics.bootstrap import (
    bootstrap_counts,
    bootstrap_ratio_difference_intervals,
    counts_digest,
    percentile_interval,
    ratio_point_and_boot,
)


def test_counts_rows_sum_to_units_and_follow_the_seed():
    counts = bootstrap_counts(7, n_resamples=50, seed=0)
    assert counts.shape == (50, 7)
    assert np.all(counts.sum(axis=1) == 7)
    assert counts_digest(counts) == counts_digest(bootstrap_counts(7, n_resamples=50, seed=0))
    assert counts_digest(counts) != counts_digest(bootstrap_counts(7, n_resamples=50, seed=1))
    with pytest.raises(ValueError):
        bootstrap_counts(0, n_resamples=5, seed=0)
    with pytest.raises(ValueError):
        bootstrap_counts(3, n_resamples=0, seed=0)


def test_counts_give_the_same_interval_as_the_direct_draws():
    rng = np.random.default_rng(3)
    den = rng.uniform(1.0, 2.0, size=12)
    num = {"a": den * rng.uniform(0.1, 0.2, size=12), "b": den * rng.uniform(0.2, 0.3, size=12)}
    direct = bootstrap_ratio_difference_intervals(
        num, {"a": den, "b": den}, [("b", "a")], n_resamples=400, seed=5
    )[("b", "a")]
    counts = bootstrap_counts(12, n_resamples=400, seed=5)
    point_a, boot_a = ratio_point_and_boot(num["a"], den, counts)
    point_b, boot_b = ratio_point_and_boot(num["b"], den, counts)
    low, high = percentile_interval(boot_b - boot_a)
    assert point_b - point_a == pytest.approx(direct["estimate"], rel=1e-12)
    assert low == pytest.approx(direct["low"], rel=1e-9)
    assert high == pytest.approx(direct["high"], rel=1e-9)


def test_ratio_is_ratio_of_sums_and_checks_shapes():
    point, boot = ratio_point_and_boot(np.array([1.0, 9.0]), np.array([1.0, 99.0]), np.array([[1.0, 1.0], [2.0, 0.0]]))
    assert point == pytest.approx(0.1)
    assert boot.tolist() == pytest.approx([0.1, 1.0])
    with pytest.raises(ValueError):
        ratio_point_and_boot(np.ones(2), np.ones(3), np.ones((1, 2)))
    with pytest.raises(ValueError):
        ratio_point_and_boot(np.ones(2), np.ones(2), np.ones((1, 3)))
    assert percentile_interval(np.arange(101.0)) == (2.5, 97.5)
    assert percentile_interval(np.arange(101.0), (5.0, 95.0)) == (5.0, 95.0)
