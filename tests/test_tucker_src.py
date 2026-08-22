"""Tucker / HOSVD 共通化の回帰テスト。

参照元 Notebook:
    notebooks/20_tucker/00_fundamentals/00_tucker_hosvd_basics.ipynb
    notebooks/20_tucker/00_fundamentals/01_rank_error_tradeoff.ipynb
"""

import pytest
import torch

from nn_compression.compression import (
    compression_factor,
    hosvd,
    parameter_ratio,
    reconstruct_tucker,
    tucker_parameter_count,
)
from nn_compression.metrics import relative_frobenius_error
from nn_compression.tensor import fold, mode_dot, unfold

SHAPE = (6, 5, 4)
FULL_RANKS = {0: 6, 1: 5, 2: 4}


def _tensor(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*SHAPE)


def test_fold_unfold_roundtrip_per_mode():
    X = _tensor()
    for mode in range(X.ndim):
        assert torch.equal(fold(unfold(X, mode), mode, X.shape), X)


def test_unfold_shape_per_mode():
    X = _tensor()
    for mode, dim in enumerate(SHAPE):
        unfolded = unfold(X, mode)
        assert unfolded.shape == (dim, X.numel() // dim)


def test_mode_dot_shape_and_value():
    X = _tensor()
    matrix = torch.randn(3, SHAPE[1])
    Y = mode_dot(X, matrix, mode=1)
    assert Y.shape == (SHAPE[0], 3, SHAPE[2])
    # mode-n product は unfolding 上の行列積と一致する
    assert torch.allclose(unfold(Y, 1), matrix @ unfold(X, 1), atol=1e-5)


def test_mode_dot_identity_keeps_tensor():
    X = _tensor()
    identity = torch.eye(SHAPE[2])
    assert torch.allclose(mode_dot(X, identity, mode=2), X, atol=1e-6)


def test_full_rank_hosvd_reconstruction_error_is_small():
    X = _tensor()
    core, factors = hosvd(X, FULL_RANKS)
    assert tuple(core.shape) == SHAPE
    X_hat = reconstruct_tucker(core, factors)
    assert relative_frobenius_error(X, X_hat).item() < 1e-5


def test_low_rank_hosvd_reconstructs_original_shape_with_larger_error():
    X = _tensor()
    full_error = relative_frobenius_error(
        X, reconstruct_tucker(*hosvd(X, FULL_RANKS))
    ).item()

    low_ranks = {0: 3, 1: 2, 2: 2}
    core, factors = hosvd(X, low_ranks)
    assert tuple(core.shape) == (3, 2, 2)

    X_hat = reconstruct_tucker(core, factors)
    assert X_hat.shape == X.shape
    assert relative_frobenius_error(X, X_hat).item() > full_error


def test_partial_hosvd_keeps_unspecified_mode_dimension():
    X = _tensor()
    ranks = {0: 3, 1: 2}
    core, factors = hosvd(X, ranks)
    # mode 2 は未指定なので元の dimension が core に残る
    assert tuple(core.shape) == (3, 2, SHAPE[2])
    assert sorted(factors) == [0, 1]
    assert reconstruct_tucker(core, factors).shape == X.shape


def test_factor_matrix_shapes():
    X = _tensor()
    ranks = {0: 3, 1: 2, 2: 2}
    _, factors = hosvd(X, ranks)
    for mode, rank in ranks.items():
        assert factors[mode].shape == (SHAPE[mode], rank)


def test_hosvd_factors_come_from_original_unfolding():
    """factor は逐次更新した core ではなく元の X の unfolding から求める。"""
    X = _tensor()
    ranks = {0: 3, 1: 2, 2: 2}
    _, factors = hosvd(X, ranks)
    for mode, rank in ranks.items():
        _, singular_values, _ = torch.linalg.svd(
            unfold(X, mode), full_matrices=False
        )
        reconstructed = factors[mode] @ factors[mode].T @ unfold(X, mode)
        expected_error = torch.linalg.vector_norm(
            unfold(X, mode) - reconstructed
        )
        tail_energy = singular_values[rank:].square().sum().sqrt()
        assert torch.allclose(expected_error, tail_energy, atol=1e-4)


def test_tucker_parameter_count_full():
    # core 3*2*2 + factors 6*3 + 5*2 + 4*2
    assert tucker_parameter_count(SHAPE, {0: 3, 1: 2, 2: 2}) == 12 + 18 + 10 + 8


def test_tucker_parameter_count_partial_keeps_original_dimension():
    # mode 2 は未指定なので core に 4 が残り、factor も持たない
    assert tucker_parameter_count(SHAPE, {0: 3, 1: 2}) == 3 * 2 * 4 + 18 + 10


def test_tucker_parameter_count_matches_actual_tensors():
    X = _tensor()
    for ranks in ({0: 3, 1: 2, 2: 2}, {0: 3, 1: 2}, {1: 2}):
        core, factors = hosvd(X, ranks)
        actual = core.numel() + sum(U.numel() for U in factors.values())
        assert tucker_parameter_count(X.shape, ranks) == actual


def test_parameter_ratio_and_compression_factor_are_reciprocal():
    ranks = {0: 3, 1: 2, 2: 2}
    ratio = parameter_ratio(SHAPE, ranks)
    factor = compression_factor(SHAPE, ranks)
    assert ratio == pytest.approx(tucker_parameter_count(SHAPE, ranks) / 120)
    assert factor == pytest.approx(1.0 / ratio)


def test_unfold_rejects_out_of_range_mode():
    X = _tensor()
    with pytest.raises(ValueError):
        unfold(X, 3)
    with pytest.raises(ValueError):
        unfold(X, -1)


def test_fold_rejects_out_of_range_mode():
    X = _tensor()
    with pytest.raises(ValueError):
        fold(unfold(X, 0), 3, X.shape)


def test_mode_dot_rejects_invalid_mode_and_matrix():
    X = _tensor()
    with pytest.raises(ValueError):
        mode_dot(X, torch.randn(2, SHAPE[0]), mode=3)
    with pytest.raises(ValueError):
        # 3 階テンソルは行列ではない
        mode_dot(X, torch.randn(2, 2, 2), mode=0)
    with pytest.raises(ValueError):
        # matrix.shape[1] が X.shape[mode] と一致しない
        mode_dot(X, torch.randn(2, SHAPE[0] + 1), mode=0)


def test_hosvd_rejects_rank_out_of_range():
    X = _tensor()
    with pytest.raises(ValueError):
        hosvd(X, {0: 0})
    with pytest.raises(ValueError):
        hosvd(X, {0: SHAPE[0] + 1})


def test_relative_frobenius_error_validations():
    X = _tensor()
    with pytest.raises(ValueError):
        relative_frobenius_error(X, torch.randn(2, 2))
    with pytest.raises(ValueError):
        zeros = torch.zeros(*SHAPE)
        relative_frobenius_error(zeros, zeros)
