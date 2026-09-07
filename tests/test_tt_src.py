"""TT-SVD (Tensor Train) 共通化の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/00_fundamentals/00_tt_svd_3way_basics.ipynb
    notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb
    notebooks/30_tt_mps/00_fundamentals/02_truncation_error_tradeoff.ipynb
"""

from __future__ import annotations

import pytest
import torch

from nn_compression.compression import (
    tt_num_parameters,
    tt_reconstruct,
    tt_svd,
    tt_svd_exact,
    tt_unfold,
)
from nn_compression.compression.tt_validation import validate_tt_cut_index
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
    torch.set_default_dtype(torch.float64)
    try:
        X = _tensor(seed=1)
        cores = tt_svd_exact(X)
        X_hat = tt_reconstruct(cores)

        assert X_hat.shape == X.shape
        assert relative_frobenius_error(X, X_hat).item() < 1e-10
    finally:
        torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# tt_reconstruct
# ---------------------------------------------------------------------------


def test_tt_reconstruct_restores_original_shape():
    X = _tensor(seed=0, shape=SHAPE_3D)
    cores = tt_svd_exact(X)
    assert tt_reconstruct(cores).shape == X.shape


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
