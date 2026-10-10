"""TT-matrix の mode 分解候補 helper（``ordered_factorizations``）の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/00_mlp_ttlinear_logits_equivalence.ipynb
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nn_compression.compression import (
    dense_to_tt_matrix_cores,
    ordered_factorizations,
    tt_matrix_num_parameters,
    tt_matrix_num_parameters_from_ranks,
    tt_matrix_physical_sizes,
    tt_matrix_ranks,
    tt_matrix_shape_imbalance,
)


def test_ordered_factorizations_keeps_order_variants_as_distinct_candidates():
    candidates = ordered_factorizations(12, 2)
    assert (2, 6) in candidates
    assert (6, 2) in candidates
    assert (3, 4) in candidates
    assert (4, 3) in candidates


def test_ordered_factorizations_exact_result_for_12_into_2():
    assert ordered_factorizations(12, 2) == [(2, 6), (3, 4), (4, 3), (6, 2)]


def test_ordered_factorizations_exact_result_for_8_into_3():
    assert ordered_factorizations(8, 3) == [(2, 2, 2)]


@pytest.mark.parametrize(
    ("value", "num_factors", "min_factor"),
    [(12, 2, 2), (64, 3, 2), (36, 3, 2), (72, 4, 2), (30, 3, 1), (16, 2, 3)],
)
def test_ordered_factorizations_products_and_bounds(value, num_factors, min_factor):
    candidates = ordered_factorizations(value, num_factors, min_factor)
    assert len(candidates) > 0
    assert len(set(candidates)) == len(candidates)  # 重複なし
    for candidate in candidates:
        assert len(candidate) == num_factors
        assert math.prod(candidate) == value
        assert all(factor >= min_factor for factor in candidate)


def test_ordered_factorizations_matches_brute_force_enumeration():
    value, num_factors, min_factor = 48, 3, 2
    expected = sorted(
        (a, b, c)
        for a in range(min_factor, value + 1)
        for b in range(min_factor, value + 1)
        for c in range(min_factor, value + 1)
        if a * b * c == value
    )
    assert ordered_factorizations(value, num_factors, min_factor) == expected


def test_ordered_factorizations_is_deterministic_and_lexicographic():
    first = ordered_factorizations(72, 3)
    second = ordered_factorizations(72, 3)
    assert first == second
    assert first == sorted(first)


def test_ordered_factorizations_does_not_embed_dataset_specific_sizes():
    # Fashion-MNIST の 784 / 512 以外の値でも一般に動く。
    assert ordered_factorizations(35, 2) == [(5, 7), (7, 5)]
    # 784 = 2^4 * 7^2、512 = 2^9 でも通常の入力として扱う。
    for candidate in ordered_factorizations(784, 3):
        assert math.prod(candidate) == 784
    for candidate in ordered_factorizations(512, 3):
        assert math.prod(candidate) == 512


def test_ordered_factorizations_single_factor():
    assert ordered_factorizations(12, 1) == [(12,)]
    assert ordered_factorizations(2, 1) == [(2,)]


def test_ordered_factorizations_single_factor_below_min_factor_is_empty():
    assert ordered_factorizations(1, 1) == []
    assert ordered_factorizations(3, 1, min_factor=4) == []
    assert ordered_factorizations(1, 1, min_factor=1) == [(1,)]


@pytest.mark.parametrize(
    ("value", "num_factors", "min_factor"),
    [
        (7, 2, 2),  # 素数は2つの2以上の因子へ分解できない
        (12, 4, 2),  # 2*2*3 までしか作れない
        (1, 2, 2),
        (6, 2, 4),
    ],
)
def test_ordered_factorizations_returns_empty_list_when_no_candidate(
    value, num_factors, min_factor
):
    assert ordered_factorizations(value, num_factors, min_factor) == []


def test_ordered_factorizations_min_factor_one_allows_unit_factors():
    assert ordered_factorizations(1, 3, min_factor=1) == [(1, 1, 1)]
    assert (1, 6) in ordered_factorizations(6, 2, min_factor=1)


def _reference_ordered_factorizations(value, num_factors, min_factor):
    """修正前と同じ再帰実装（小さい入力での候補順の基準）。"""

    def divisors(n, lower):
        return [d for d in range(1, n + 1) if n % d == 0 and d >= lower]

    def recurse(remaining, factors_left, prefix):
        if factors_left == 1:
            return [prefix + (remaining,)] if remaining >= min_factor else []
        results = []
        for factor in divisors(remaining, min_factor):
            next_remaining = remaining // factor
            if next_remaining < min_factor ** (factors_left - 1):
                continue
            results.extend(recurse(next_remaining, factors_left - 1, prefix + (factor,)))
        return results

    return recurse(value, num_factors, ())


@pytest.mark.parametrize("min_factor", [1, 2, 3])
@pytest.mark.parametrize("num_factors", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("value", [1, 2, 6, 12, 16, 30, 36, 64, 72, 96])
def test_ordered_factorizations_keeps_candidate_order_of_reference(
    value, num_factors, min_factor
):
    result = ordered_factorizations(value, num_factors, min_factor)

    assert result == _reference_ordered_factorizations(value, num_factors, min_factor)
    assert len(set(result)) == len(result)
    for candidate in result:
        assert len(candidate) == num_factors
        assert math.prod(candidate) == value
        assert all(factor >= min_factor for factor in candidate)


def test_ordered_factorizations_pins_candidate_order():
    assert ordered_factorizations(12, 3) == [(2, 2, 3), (2, 3, 2), (3, 2, 2)]
    assert ordered_factorizations(8, 2, min_factor=1) == [
        (1, 8),
        (2, 4),
        (4, 2),
        (8, 1),
    ]


def test_ordered_factorizations_does_not_hit_recursion_limit_for_many_unit_factors():
    result = ordered_factorizations(1, 1000, min_factor=1)

    assert len(result) == 1
    assert len(result[0]) == 1000
    assert all(factor == 1 for factor in result[0])


def test_ordered_factorizations_handles_many_factors_with_a_single_non_unit_factor():
    # 深さ 1000 の探索で、2 を置く位置だけ異なる 1000 候補。
    result = ordered_factorizations(2, 1000, min_factor=1)

    assert len(result) == 1000
    assert len(set(result)) == 1000
    for candidate in result:
        assert len(candidate) == 1000
        assert math.prod(candidate) == 2
        assert sorted(candidate)[-1] == 2
    # 先頭因子の昇順: 1 が先に並ぶものから、2 が先頭に来る候補が最後。
    assert result[0] == (1,) * 999 + (2,)
    assert result[-1] == (2,) + (1,) * 999
    assert result == sorted(result)


def test_ordered_factorizations_handles_deep_valid_input_with_min_factor_two():
    # 2**30 を 30 個の因子へ: 候補は (2, ..., 2) の 1 件だけ。
    result = ordered_factorizations(2**30, 30)
    assert result == [(2,) * 30]


def test_ordered_factorizations_huge_num_factors_has_no_candidate_and_is_fast():
    # min_factor>=2 では、積が value 以下に収まる個数に上限がある。
    assert ordered_factorizations(12, 10**6) == []
    assert ordered_factorizations(10**18, 10**6, min_factor=3) == []


def test_ordered_factorizations_accepts_numpy_integers():
    assert ordered_factorizations(np.int64(12), np.int64(2)) == [
        (2, 6),
        (3, 4),
        (4, 3),
        (6, 2),
    ]


@pytest.mark.parametrize("bad", [0, -1, -12])
@pytest.mark.parametrize("argument", ["value", "num_factors", "min_factor"])
def test_ordered_factorizations_rejects_non_positive_values(argument, bad):
    kwargs = {"value": 12, "num_factors": 2, "min_factor": 2}
    kwargs[argument] = bad
    with pytest.raises(ValueError):
        ordered_factorizations(**kwargs)


@pytest.mark.parametrize("bad", [True, False])
@pytest.mark.parametrize("argument", ["value", "num_factors", "min_factor"])
def test_ordered_factorizations_rejects_bool(argument, bad):
    kwargs = {"value": 12, "num_factors": 2, "min_factor": 2}
    kwargs[argument] = bad
    with pytest.raises(TypeError):
        ordered_factorizations(**kwargs)


@pytest.mark.parametrize("bad", [2.0, 1.5, "2", None])
@pytest.mark.parametrize("argument", ["value", "num_factors", "min_factor"])
def test_ordered_factorizations_rejects_non_integer(argument, bad):
    kwargs = {"value": 12, "num_factors": 2, "min_factor": 2}
    kwargs[argument] = bad
    with pytest.raises(TypeError):
        ordered_factorizations(**kwargs)


# ---------------------------------------------------------------------------
# tt_matrix_physical_sizes / tt_matrix_shape_imbalance
# （参照元: notebooks/.../00_mlp_ttlinear_logits_equivalence.ipynb の候補評価 helper）
# ---------------------------------------------------------------------------


def test_tt_matrix_physical_sizes_hand_computed():
    assert tt_matrix_physical_sizes((4, 7, 28), (4, 4, 49)) == (16, 28, 1372)
    assert tt_matrix_physical_sizes((6,), (4,)) == (24,)


def test_tt_matrix_physical_sizes_returns_python_ints_and_accepts_numpy_ints():
    # 既存 API と同じ契約: NumPy 整数は「要素」として受け付ける（ndarray 自体は列として扱わない）。
    sizes = tt_matrix_physical_sizes([np.int64(2), np.int32(3)], (np.int64(5), 7))

    assert sizes == (10, 21)
    assert all(type(size) is int for size in sizes)


def test_tt_matrix_physical_sizes_do_not_mutate_inputs():
    out_modes, in_modes = [2, 3], [5, 7]

    tt_matrix_physical_sizes(out_modes, in_modes)

    assert out_modes == [2, 3] and in_modes == [5, 7]


def test_tt_matrix_shape_imbalance_hand_computed():
    assert tt_matrix_shape_imbalance((2, 2), (2, 2)) == 1.0  # 4, 4
    assert tt_matrix_shape_imbalance((2, 4), (2, 4)) == 4.0  # 4, 16
    assert tt_matrix_shape_imbalance((6,), (4,)) == 1.0  # d=1 は常に 1


def test_tt_matrix_shape_imbalance_is_independent_of_site_order():
    assert tt_matrix_shape_imbalance((2, 4), (2, 4)) == tt_matrix_shape_imbalance((4, 2), (4, 2))


@pytest.mark.parametrize(
    "function", [tt_matrix_physical_sizes, tt_matrix_shape_imbalance]
)
def test_tt_matrix_mode_helpers_validate_modes(function):
    with pytest.raises(ValueError):
        function((2, 3), (2,))  # 長さ不一致
    with pytest.raises(ValueError):
        function((), ())
    with pytest.raises(ValueError):
        function((2, 0), (2, 2))
    with pytest.raises(TypeError):
        function((2, True), (2, 2))
    with pytest.raises(TypeError):
        function((2, 2.0), (2, 2))


# ---------------------------------------------------------------------------
# tt_matrix_num_parameters_from_ranks
# ---------------------------------------------------------------------------


def test_tt_matrix_num_parameters_from_ranks_hand_computed():
    # 2*2*1*... : r=(1,2,2,1), 各 site 2x2 -> 1*4*2 + 2*4*2 + 2*4*1 = 8 + 16 + 8 = 32
    assert tt_matrix_num_parameters_from_ranks((2, 2, 2), (2, 2, 2), (1, 2, 2, 1)) == 32
    # Notebook 07 の rank 1 / full rank
    assert tt_matrix_num_parameters_from_ranks((2, 2, 2), (2, 2, 2), (1, 1, 1, 1)) == 12
    assert tt_matrix_num_parameters_from_ranks((2, 2, 2), (2, 2, 2), (1, 4, 4, 1)) == 96


def test_tt_matrix_num_parameters_from_ranks_d1_is_dense_size():
    assert tt_matrix_num_parameters_from_ranks((6,), (4,), (1, 1)) == 24


def test_tt_matrix_num_parameters_from_ranks_can_exceed_dense_parameters():
    assert tt_matrix_num_parameters_from_ranks((2, 2, 2), (2, 2, 2), (1, 4, 4, 1)) > 8 * 8


@pytest.mark.parametrize(
    ("out_modes", "in_modes"),
    [
        ((6,), (4,)),
        ((2, 3), (3, 2)),
        ((2, 2, 2), (2, 2, 2)),
        ((2, 2, 2, 2), (2, 2, 2, 2)),
        ((2, 3, 2), (3, 2, 2)),
    ],
)
@pytest.mark.parametrize("max_rank", [None, 1, 2, 3, 100])
def test_tt_matrix_num_parameters_from_ranks_equals_real_core_element_count(
    out_modes, in_modes, max_rank
):
    generator = torch.Generator().manual_seed(0)
    W = torch.randn(
        math.prod(out_modes), math.prod(in_modes), generator=generator, dtype=torch.float64
    )
    cores = dense_to_tt_matrix_cores(W, out_modes, in_modes, max_rank=max_rank)

    predicted = tt_matrix_num_parameters_from_ranks(out_modes, in_modes, tt_matrix_ranks(cores))

    assert predicted == tt_matrix_num_parameters(cores)
    assert predicted == sum(core.numel() for core in cores)


def test_tt_matrix_num_parameters_from_ranks_returns_python_int_and_accepts_numpy_ints():
    result = tt_matrix_num_parameters_from_ranks(
        [np.int64(2), 2], (2, np.int32(2)), [np.int64(1), np.int64(3), np.int64(1)]
    )

    assert result == 1 * 4 * 3 + 3 * 4 * 1
    assert type(result) is int


def test_tt_matrix_num_parameters_from_ranks_accepts_tuple_and_list_ranks():
    assert tt_matrix_num_parameters_from_ranks([2, 2], [2, 2], [1, 2, 1]) == 16
    assert tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), (1, 2, 1)) == 16


def test_tt_matrix_num_parameters_from_ranks_rejects_wrong_length():
    with pytest.raises(ValueError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), (1, 2))
    with pytest.raises(ValueError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), (1, 2, 2, 1))
    with pytest.raises(ValueError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), ())


def test_tt_matrix_num_parameters_from_ranks_rejects_mismatched_mode_lengths():
    with pytest.raises(ValueError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2, 2), (1, 2, 1))


@pytest.mark.parametrize("ranks", [(2, 2, 1), (1, 2, 2), (3, 2, 3)])
def test_tt_matrix_num_parameters_from_ranks_rejects_non_unit_boundary_rank(ranks):
    with pytest.raises(ValueError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), ranks)


@pytest.mark.parametrize("ranks", [(1, 0, 1), (1, -2, 1), (0, 2, 1)])
def test_tt_matrix_num_parameters_from_ranks_rejects_non_positive_rank(ranks):
    with pytest.raises(ValueError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), ranks)


@pytest.mark.parametrize("ranks", [(1, True, 1), (True, 2, 1), (1, 2.0, 1), (1, "2", 1), (1, None, 1)])
def test_tt_matrix_num_parameters_from_ranks_rejects_non_integer_or_bool_rank(ranks):
    with pytest.raises(TypeError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), ranks)


@pytest.mark.parametrize("ranks", [5, "1,2,1", None, 1.0])
def test_tt_matrix_num_parameters_from_ranks_rejects_non_sequence_ranks(ranks):
    with pytest.raises(TypeError):
        tt_matrix_num_parameters_from_ranks((2, 2), (2, 2), ranks)


def test_tt_matrix_num_parameters_from_ranks_rejects_invalid_modes():
    with pytest.raises(ValueError):
        tt_matrix_num_parameters_from_ranks((2, 0), (2, 2), (1, 2, 1))
    with pytest.raises(TypeError):
        tt_matrix_num_parameters_from_ranks((2, True), (2, 2), (1, 2, 1))
