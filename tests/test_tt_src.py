"""TT-SVD (Tensor Train) 共通化の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/00_fundamentals/00_tt_svd_3way_basics.ipynb
    notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb
    notebooks/30_tt_mps/00_fundamentals/02_truncation_error_tradeoff.ipynb
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from nn_compression.compression import (
    tt_num_parameters,
    tt_physical_shape,
    tt_ranks,
    tt_reconstruct,
    tt_svd,
    tt_svd_error_bound,
    tt_svd_exact,
    tt_svd_ranks,
    tt_unfold,
)
from nn_compression.compression.tt_validation import (
    validate_tt_cores,
    validate_tt_cut_index,
)
from nn_compression.metrics import relative_frobenius_error

SHAPE_4D = (2, 3, 2, 2)  # notebook 01 と同じ 4 階テンソル
SHAPE_3D = (2, 3, 4)  # notebook 00 と同じ 3 階テンソル


def _tensor(seed: int, shape: tuple[int, ...] = SHAPE_4D) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# validate_tt_cut_index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_validate_tt_cut_index_accepts_valid_range(k):
    validate_tt_cut_index(k, ndim=4)  # 例外が出なければOK


def test_validate_tt_cut_index_rejects_k_zero():
    with pytest.raises(ValueError):
        validate_tt_cut_index(0, ndim=4)


def test_validate_tt_cut_index_rejects_k_equal_ndim():
    with pytest.raises(ValueError):
        validate_tt_cut_index(4, ndim=4)


@pytest.mark.parametrize("k", [True, False])
def test_validate_tt_cut_index_rejects_bool(k):
    with pytest.raises(TypeError):
        validate_tt_cut_index(k, ndim=4)


@pytest.mark.parametrize("k", [1.5, "1", None])
def test_validate_tt_cut_index_rejects_non_integer(k):
    with pytest.raises(TypeError):
        validate_tt_cut_index(k, ndim=4)


# ---------------------------------------------------------------------------
# tt_unfold
# ---------------------------------------------------------------------------


def test_tt_unfold_shapes_for_4d_tensor():
    X = _tensor(seed=1)
    n1, n2, n3, n4 = X.shape

    assert tt_unfold(X, 1).shape == (n1, n2 * n3 * n4)
    assert tt_unfold(X, 2).shape == (n1 * n2, n3 * n4)
    assert tt_unfold(X, 3).shape == (n1 * n2 * n3, n4)


def test_tt_unfold_rejects_cut_out_of_range():
    X = _tensor(seed=1)
    with pytest.raises(ValueError):
        tt_unfold(X, 0)
    with pytest.raises(ValueError):
        tt_unfold(X, X.ndim)


def test_tt_unfold_element_order_matches_row_major_layout():
    """shapeだけでなく、要素順（reshapeの並べ替え規約）の回帰も検出する。

    先頭 k 軸を row 側、残りを column 側へ C-order（row-major）でまとめた
    行列を ``tt_unfold`` を使わずに素朴な多重ループで独立に構築し、
    ``tt_unfold(X, k)`` の出力と要素単位で一致することを確認する。
    """
    shape = (2, 3, 4)
    X = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)

    for k in range(1, len(shape)):
        left_shape = shape[:k]
        right_shape = shape[k:]
        left_dim = math.prod(left_shape)
        right_dim = math.prod(right_shape)
        expected = torch.empty(left_dim, right_dim, dtype=X.dtype)

        for left_idx in itertools.product(*(range(s) for s in left_shape)):
            row = 0
            for size, i in zip(left_shape, left_idx):
                row = row * size + i
            for right_idx in itertools.product(*(range(s) for s in right_shape)):
                col = 0
                for size, j in zip(right_shape, right_idx):
                    col = col * size + j
                expected[row, col] = X[left_idx + right_idx]

        actual = tt_unfold(X, k)
        assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# tt_svd_exact / tt_svd: dtype validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_svd_exact_accepts_supported_dtypes(dtype):
    X = torch.randn(2, 3, 4, dtype=dtype)
    cores = tt_svd_exact(X)  # 例外が出なければOK
    assert cores[0].dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tt_svd_accepts_supported_dtypes(dtype):
    X = torch.randn(2, 3, 4, dtype=dtype)
    cores = tt_svd(X, max_rank=2)  # 例外が出なければOK
    assert cores[0].dtype == dtype


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ],
)
def test_tt_svd_exact_rejects_unsupported_dtypes(dtype):
    X = torch.zeros(2, 3, 4, dtype=dtype)
    with pytest.raises(TypeError):
        tt_svd_exact(X)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ],
)
def test_tt_svd_rejects_unsupported_dtypes(dtype):
    X = torch.zeros(2, 3, 4, dtype=dtype)
    with pytest.raises(TypeError):
        tt_svd(X, max_rank=2)


# ---------------------------------------------------------------------------
# 零テンソル（数値rank 0）の特殊ケース
# ---------------------------------------------------------------------------


def test_tt_svd_exact_handles_zero_tensor():
    X = torch.zeros(2, 3, 4, dtype=torch.float64)
    cores = tt_svd_exact(X)  # 例外にならないこと

    # 数値rankが0でも、bond dimensionは最低1を使う
    bond_ranks = [core.shape[-1] for core in cores[:-1]]
    assert bond_ranks == [1] * (X.ndim - 1)

    X_hat = tt_reconstruct(cores)
    assert X_hat.shape == X.shape
    assert torch.allclose(X_hat, torch.zeros_like(X))
    # X が零テンソルのときは relative_frobenius_error の分母が0になるため
    # （0除算のためValueErrorとして拒否される契約）、絶対誤差で確認する。
    absolute_error = torch.linalg.vector_norm(X - X_hat).item()
    assert absolute_error == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("max_rank", [1, 2, 4])
def test_tt_svd_handles_zero_tensor(max_rank):
    X = torch.zeros(2, 3, 4, dtype=torch.float64)
    cores = tt_svd(X, max_rank)  # 例外にならないこと

    bond_ranks = [core.shape[-1] for core in cores[:-1]]
    # 数値rankが0なので、max_rankに関わらずbond dimensionは1で頭打ちになる
    assert bond_ranks == [1] * (X.ndim - 1)

    X_hat = tt_reconstruct(cores)
    assert X_hat.shape == X.shape
    assert torch.allclose(X_hat, torch.zeros_like(X))


# ---------------------------------------------------------------------------
# tt_svd_exact
# ---------------------------------------------------------------------------


def test_tt_svd_exact_core_boundary_ranks_are_one():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)

    assert len(cores) == X.ndim
    assert cores[0].shape[0] == 1
    assert cores[-1].shape[-1] == 1


def test_tt_svd_exact_bond_rank_matches_cut_unfolding_rank():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)

    for k in range(1, X.ndim):
        expected_rank = int(torch.linalg.matrix_rank(tt_unfold(X, k)).item())
        assert cores[k - 1].shape[-1] == expected_rank


def test_tt_svd_exact_reconstructs_within_float64_tolerance():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        X = _tensor(seed=1)
        cores = tt_svd_exact(X)
        X_hat = tt_reconstruct(cores)

        assert X_hat.shape == X.shape
        assert relative_frobenius_error(X, X_hat).item() < 1e-10
    finally:
        torch.set_default_dtype(old_dtype)


# ---------------------------------------------------------------------------
# tt_reconstruct
# ---------------------------------------------------------------------------


def test_tt_reconstruct_restores_original_shape():
    X = _tensor(seed=0, shape=SHAPE_3D)
    cores = tt_svd_exact(X)
    assert tt_reconstruct(cores).shape == X.shape


# ---------------------------------------------------------------------------
# validate_tt_cores（tt_reconstruct / tt_num_parameters が使う structure validation）
# ---------------------------------------------------------------------------


def _valid_cores(dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    """(1, 2, 2) -> (2, 3, 4) -> (4, 4, 1) という有効な3-core TT。"""
    return [
        torch.randn(1, 2, 2, dtype=dtype),
        torch.randn(2, 3, 4, dtype=dtype),
        torch.randn(4, 4, 1, dtype=dtype),
    ]


def test_validate_tt_cores_accepts_valid_core_list():
    validate_tt_cores(_valid_cores())  # 例外が出なければOK


def test_validate_tt_cores_rejects_empty_list():
    with pytest.raises(ValueError):
        validate_tt_cores([])


def test_validate_tt_cores_rejects_non_3d_core():
    cores = _valid_cores()
    cores[1] = cores[1].reshape(2, 3 * 4)  # 2階へ潰す
    with pytest.raises(ValueError):
        validate_tt_cores(cores)


def test_validate_tt_cores_rejects_adjacent_bond_mismatch():
    cores = _valid_cores()
    cores[1] = torch.randn(3, 3, 4, dtype=cores[1].dtype)  # left bondを2→3にずらす
    with pytest.raises(ValueError):
        validate_tt_cores(cores)


def test_validate_tt_cores_rejects_left_boundary_bond_not_one():
    cores = _valid_cores()
    cores[0] = torch.randn(2, 2, 2, dtype=cores[0].dtype)
    with pytest.raises(ValueError):
        validate_tt_cores(cores)


def test_validate_tt_cores_rejects_right_boundary_bond_not_one():
    cores = _valid_cores()
    cores[-1] = torch.randn(4, 4, 2, dtype=cores[-1].dtype)
    with pytest.raises(ValueError):
        validate_tt_cores(cores)


def test_validate_tt_cores_rejects_dtype_mismatch():
    cores = _valid_cores(dtype=torch.float64)
    cores[1] = cores[1].to(torch.float32)
    with pytest.raises(TypeError):
        validate_tt_cores(cores)


def test_validate_tt_cores_rejects_non_tensor_element():
    cores = _valid_cores()
    cores[1] = cores[1].tolist()
    with pytest.raises(TypeError):
        validate_tt_cores(cores)


def test_tt_reconstruct_rejects_empty_core_list():
    with pytest.raises(ValueError):
        tt_reconstruct([])


def test_tt_reconstruct_rejects_invalid_core_list():
    cores = _valid_cores()
    cores[1] = torch.randn(3, 3, 4, dtype=cores[1].dtype)
    with pytest.raises(ValueError):
        tt_reconstruct(cores)


def test_tt_num_parameters_rejects_empty_core_list():
    with pytest.raises(ValueError):
        tt_num_parameters([])


# ---------------------------------------------------------------------------
# tt_svd（rank上限付き）
# ---------------------------------------------------------------------------


def test_tt_svd_bond_ranks_are_bounded_by_max_rank():
    X = _tensor(seed=2, shape=(4, 4, 4, 4))
    max_rank = 2
    cores = tt_svd(X, max_rank)

    for core in cores[:-1]:
        assert core.shape[-1] <= max_rank


def test_tt_svd_with_smaller_max_rank_increases_reconstruction_error():
    X = _tensor(seed=2, shape=(4, 4, 4, 4))

    loose_error = relative_frobenius_error(
        X, tt_reconstruct(tt_svd(X, 4))
    ).item()
    tight_error = relative_frobenius_error(
        X, tt_reconstruct(tt_svd(X, 1))
    ).item()

    assert tight_error >= loose_error


def test_tt_svd_matches_exact_when_max_rank_large_enough():
    """SVDの符号非一意性があるため、core tensorの完全一致は要求しない。

    bond ranks・再構成テンソル・再構成誤差がexact版と整合することを確認する。
    """
    X = _tensor(seed=1)
    exact_cores = tt_svd_exact(X)

    large_max_rank = max(X.shape) ** 2  # 実際に出うるbondより十分大きい
    truncated_cores = tt_svd(X, large_max_rank)

    exact_bond_ranks = [core.shape[-1] for core in exact_cores[:-1]]
    truncated_bond_ranks = [core.shape[-1] for core in truncated_cores[:-1]]
    assert exact_bond_ranks == truncated_bond_ranks

    X_hat_exact = tt_reconstruct(exact_cores)
    X_hat_truncated = tt_reconstruct(truncated_cores)
    assert torch.allclose(X_hat_exact, X_hat_truncated, atol=1e-5)

    exact_error = relative_frobenius_error(X, X_hat_exact).item()
    truncated_error = relative_frobenius_error(X, X_hat_truncated).item()
    assert truncated_error == pytest.approx(exact_error, abs=1e-5)


@pytest.mark.parametrize("max_rank", [0, -1])
def test_tt_svd_rejects_non_positive_max_rank(max_rank):
    X = _tensor(seed=1)
    with pytest.raises(ValueError):
        tt_svd(X, max_rank)


@pytest.mark.parametrize("max_rank", [True, False])
def test_tt_svd_rejects_bool_max_rank(max_rank):
    X = _tensor(seed=1)
    with pytest.raises(TypeError):
        tt_svd(X, max_rank)


@pytest.mark.parametrize("max_rank", [1.5, "2", None])
def test_tt_svd_rejects_non_integer_max_rank(max_rank):
    X = _tensor(seed=1)
    with pytest.raises(TypeError):
        tt_svd(X, max_rank)


def test_tt_svd_ranks_uses_per_bond_caps():
    X = _tensor(seed=2, shape=(4, 4, 4, 4))
    ranks = [2, 4, 2]
    cores = tt_svd_ranks(X, ranks)

    bond_ranks = [core.shape[-1] for core in cores[:-1]]
    assert bond_ranks == ranks
    assert cores[-1].shape[-1] == 1

    X_hat = tt_reconstruct(cores)
    assert X_hat.shape == X.shape


def test_tt_svd_ranks_matches_uniform_tt_svd():
    X = _tensor(seed=2, shape=(4, 4, 4, 4))
    uniform = tt_svd(X, max_rank=2)
    per_bond = tt_svd_ranks(X, [2, 2, 2])

    assert torch.allclose(tt_reconstruct(uniform), tt_reconstruct(per_bond))


def test_tt_svd_ranks_caps_at_numerical_rank():
    X = _tensor(seed=1, shape=(2, 3, 2))
    exact_bonds = [core.shape[-1] for core in tt_svd_exact(X)[:-1]]
    cores = tt_svd_ranks(X, [10**6, 10**6])
    assert [core.shape[-1] for core in cores[:-1]] == exact_bonds


def test_tt_svd_ranks_rejects_wrong_length():
    X = _tensor(seed=1, shape=(4, 4, 4, 4))
    with pytest.raises(ValueError):
        tt_svd_ranks(X, [2, 4])


@pytest.mark.parametrize("ranks", [[0, 2, 2], [2, -1, 2]])
def test_tt_svd_ranks_rejects_non_positive_rank(ranks):
    X = _tensor(seed=1, shape=(4, 4, 4, 4))
    with pytest.raises(ValueError):
        tt_svd_ranks(X, ranks)


@pytest.mark.parametrize("ranks", [2, True, "2"])
def test_tt_svd_ranks_rejects_non_sequence(ranks):
    X = _tensor(seed=1, shape=(4, 4, 4, 4))
    with pytest.raises(TypeError):
        tt_svd_ranks(X, ranks)


# ---------------------------------------------------------------------------
# tt_num_parameters
# ---------------------------------------------------------------------------


def test_tt_num_parameters_matches_numel_sum():
    X = _tensor(seed=0, shape=SHAPE_3D)
    cores = tt_svd_exact(X)
    expected = sum(core.numel() for core in cores)
    assert tt_num_parameters(cores) == expected


def test_tt_num_parameters_matches_bond_rank_formula():
    """P_TT = sum_k r_{k-1} * n_k * r_k を core shape から独立に計算して確認する。"""
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)

    expected_from_shapes = sum(
        core.shape[0] * core.shape[1] * core.shape[2] for core in cores
    )
    assert tt_num_parameters(cores) == expected_from_shapes


# ---------------------------------------------------------------------------
# tt_physical_shape / tt_ranks
# ---------------------------------------------------------------------------


def test_tt_physical_shape_returns_middle_axis_of_each_core():
    X = _tensor(seed=0, shape=SHAPE_3D)
    cores = tt_svd_exact(X)

    assert tt_physical_shape(cores) == SHAPE_3D


def test_tt_physical_shape_matches_valid_cores_helper():
    cores = _valid_cores()
    assert tt_physical_shape(cores) == (2, 3, 4)


def test_tt_physical_shape_rejects_empty_list():
    with pytest.raises(ValueError):
        tt_physical_shape([])


def test_tt_physical_shape_rejects_invalid_core_list():
    cores = _valid_cores()
    cores[1] = torch.randn(3, 3, 4, dtype=cores[1].dtype)  # bond不整合
    with pytest.raises(ValueError):
        tt_physical_shape(cores)


def test_tt_ranks_includes_boundary_ranks_of_one():
    cores = _valid_cores()
    assert tt_ranks(cores) == (1, 2, 4, 1)


def test_tt_ranks_matches_bond_shapes_from_tt_svd_exact():
    X = _tensor(seed=1)
    cores = tt_svd_exact(X)

    expected = (1,) + tuple(core.shape[-1] for core in cores[:-1]) + (1,)
    assert tt_ranks(cores) == expected
    assert len(tt_ranks(cores)) == len(cores) + 1


def test_tt_ranks_rejects_empty_list():
    with pytest.raises(ValueError):
        tt_ranks([])


def test_tt_ranks_rejects_invalid_core_list():
    cores = _valid_cores()
    cores[0] = torch.randn(2, 2, 2, dtype=cores[0].dtype)  # 左境界rankが1でない
    with pytest.raises(ValueError):
        tt_ranks(cores)


# ---------------------------------------------------------------------------
# tt_svd_error_bound
# ---------------------------------------------------------------------------


def test_tt_svd_error_bound_matches_independent_sequential_tail_calculation():
    X = _tensor(seed=9, shape=(3, 4, 2)).double()
    ranks = [2, 1]

    bound, local_errors = tt_svd_error_bound(X, ranks)

    expected_local = []
    remainder = X
    r_left = 1
    for mode, rank_cap in enumerate(ranks):
        mat = remainder.reshape(r_left * X.shape[mode], -1)
        numerical_rank = max(1, int(torch.linalg.matrix_rank(mat).item()))
        rank = min(rank_cap, numerical_rank)
        _, singular_values, Vh = torch.linalg.svd(mat, full_matrices=False)
        expected_local.append(torch.linalg.vector_norm(singular_values[rank:]))
        remainder = (singular_values[:rank].unsqueeze(1) * Vh[:rank]).reshape(
            rank,
            *X.shape[mode + 1 :],
        )
        r_left = rank
    expected_bound = torch.linalg.vector_norm(torch.stack(expected_local))

    assert len(local_errors) == len(ranks)
    for actual, expected in zip(local_errors, expected_local):
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(bound, expected_bound, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "seed,ranks",
    [
        (10, [2, 2, 1]),
        (11, [1, 1, 1]),
        (12, [3, 2, 2]),
        (13, [2, 3, 1]),
        (14, [1, 3, 2]),
    ],
)
def test_tt_svd_error_bound_bounds_measured_tt_svd_error(seed, ranks):
    X = _tensor(seed=seed, shape=(3, 4, 3, 2)).double()

    bound, _ = tt_svd_error_bound(X, ranks)
    approximation = tt_reconstruct(tt_svd_ranks(X, ranks))
    measured_error = torch.linalg.vector_norm(X - approximation)

    assert measured_error <= bound + 1e-10


def test_tt_svd_error_bound_includes_numerical_rank_truncation():
    """指定rankより数値rankが小さい場合も、破棄成分を上限へ含める。"""
    X = torch.zeros(100, 100, dtype=torch.float64)
    X[0, 0] = 1e8
    X[1, 1] = 1e-6

    bound, local_errors = tt_svd_error_bound(X, [100])
    approximation = tt_reconstruct(tt_svd_ranks(X, [100]))
    measured_error = torch.linalg.vector_norm(X - approximation)

    assert torch.linalg.matrix_rank(X).item() == 1
    assert local_errors[0].item() > 0.0
    assert torch.equal(bound, local_errors[0])  # 1 cutなので全体上限と局所tailは同じ
    assert measured_error.item() == pytest.approx(1e-6, rel=1e-12)
    assert measured_error <= bound


def test_tt_svd_error_bound_full_cut_ranks_are_zero_and_preserve_input():
    X = _tensor(seed=11, shape=(2, 3, 2)).double()
    saved = X.clone()

    bound, local_errors = tt_svd_error_bound(X, [100, 100])

    assert bound.item() == 0.0
    assert [error.item() for error in local_errors] == [0.0, 0.0]
    assert bound.dtype == X.dtype
    assert bound.device == X.device
    assert torch.equal(X, saved)


def test_tt_svd_error_bound_preserves_autograd_graph():
    X = _tensor(seed=18, shape=(2, 3, 2)).double().requires_grad_()

    bound, local_errors = tt_svd_error_bound(X, [1, 1])
    (bound + torch.stack(local_errors).sum()).backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


@pytest.mark.parametrize("ranks", [[1], [1, 1, 1]])
def test_tt_svd_error_bound_rejects_wrong_rank_count(ranks):
    X = _tensor(seed=12, shape=(2, 3, 2)).double()
    with pytest.raises(ValueError):
        tt_svd_error_bound(X, ranks)


@pytest.mark.parametrize("ranks", [[0, 1], [1, True], 2, "1"])
def test_tt_svd_error_bound_rejects_invalid_ranks(ranks):
    X = _tensor(seed=12, shape=(2, 3, 2)).double()
    with pytest.raises((TypeError, ValueError)):
        tt_svd_error_bound(X, ranks)
