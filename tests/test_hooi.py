"""HOOI / partial Tucker-2 の回帰テスト。

参照元 Notebook:
    notebooks/20_tucker/00_fundamentals/02_hooi.ipynb
    notebooks/20_tucker/10_cifar10_cnn/04_hooi_tucker2.ipynb
"""

import pytest
import torch

from nn_compression.compression import (
    core_from_factors,
    has_converged,
    hooi,
    hooi_sweep,
    hosvd,
    reconstruct_tucker,
    tucker2_hooi,
    tucker2_hooi_sweep,
)
from nn_compression.metrics import relative_frobenius_error

SHAPE_3D = (6, 5, 4)
RANKS_3D = {0: 3, 1: 2, 2: 2}


def _tensor_3d(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*SHAPE_3D)


def _conv_weight(
    *,
    out_channels: int = 64,
    in_channels: int = 32,
    kernel_size: int = 3,
) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(out_channels, in_channels, kernel_size, kernel_size)


def test_hooi_reconstruction_error_not_worse_than_hosvd():
    """seed 固定 random tensor で HOOI 誤差が HOSVD 初期値より悪化しない。"""
    X = _tensor_3d()
    core_h, factors_h = hosvd(X, RANKS_3D)
    err_h = relative_frobenius_error(
        X, reconstruct_tucker(core_h, factors_h)
    ).item()

    _, _, history = hooi(X, RANKS_3D, max_iter=20)
    assert history[0] == pytest.approx(err_h, rel=1e-5)
    assert history[-1] <= history[0] + 1e-6


def test_hooi_works_on_three_mode_tensor():
    """3 階テンソルで ranks={0,1,2} の HOOI が動作する。"""
    X = _tensor_3d()
    core, factors, history = hooi(X, RANKS_3D, max_iter=10)

    assert tuple(core.shape) == (3, 2, 2)
    for mode, rank in RANKS_3D.items():
        assert factors[mode].shape == (SHAPE_3D[mode], rank)

    X_hat = reconstruct_tucker(core, factors)
    assert X_hat.shape == X.shape
    assert len(history) >= 2
    assert relative_frobenius_error(X, X_hat).item() == pytest.approx(
        history[-1], rel=1e-5
    )


def test_partial_hooi_works_on_conv_weight():
    """4 階 Conv weight で ranks={0,1} の partial HOOI が動作する。"""
    weight = _conv_weight()
    ranks = {0: 32, 1: 16}

    core, factors, history = hooi(weight, ranks, max_iter=10)

    k_h, k_w = weight.shape[2], weight.shape[3]
    assert tuple(core.shape) == (32, 16, k_h, k_w)
    assert factors[0].shape == (64, 32)
    assert factors[1].shape == (32, 16)

    X_hat = reconstruct_tucker(core, factors)
    assert X_hat.shape == weight.shape
    assert history[-1] <= history[0] + 1e-6


def test_tucker2_hooi_core_shape_is_rank_out_rank_in_kh_kw():
    weight = _conv_weight(out_channels=64, in_channels=32, kernel_size=3)
    rank_out, rank_in = 32, 16

    core, factors, history = tucker2_hooi(
        weight, rank_out, rank_in, max_iter=10
    )
    u_out, u_in = factors[0], factors[1]

    assert tuple(core.shape) == (rank_out, rank_in, 3, 3)
    assert u_out.shape == (64, rank_out)
    assert u_in.shape == (32, rank_in)
    assert len(history) >= 1
    assert history[-1] <= history[0] + 1e-6


def test_hooi_factor_shapes_match_ranks():
    X = _tensor_3d()
    _, factors, _ = hooi(X, RANKS_3D, max_iter=5)
    for mode, rank in RANKS_3D.items():
        assert factors[mode].shape == (SHAPE_3D[mode], rank)


def test_hooi_sweep_does_not_mutate_input_factors():
    X = _tensor_3d()
    _, factors = hosvd(X, RANKS_3D)
    snapshot = {m: U.clone() for m, U in factors.items()}

    updated = hooi_sweep(X, factors, RANKS_3D)

    for mode, U in factors.items():
        assert torch.equal(U, snapshot[mode])
    assert updated is not factors
    for mode in RANKS_3D:
        assert updated[mode] is not factors[mode]
        assert updated[mode].shape == (SHAPE_3D[mode], RANKS_3D[mode])


def test_hooi_error_history_decreases_or_converges():
    X = _tensor_3d()
    _, _, history = hooi(X, RANKS_3D, max_iter=20)

    assert history[0] >= history[-1]
    for prev, curr in zip(history, history[1:]):
        assert curr <= prev + 1e-6


def test_hooi_rejects_mismatched_factor_keys():
    X = _tensor_3d()
    _, factors = hosvd(X, {0: 3, 1: 2})
    with pytest.raises(ValueError, match="mode=2"):
        hooi_sweep(X, factors, RANKS_3D)


def test_hooi_rejects_invalid_mode():
    X = _tensor_3d()
    with pytest.raises(ValueError, match="mode="):
        hooi(X, {3: 2}, max_iter=1)


def test_hooi_rejects_invalid_mode_in_sweep():
    X = _tensor_3d()
    _, factors = hosvd(X, RANKS_3D)
    with pytest.raises(ValueError):
        hooi_sweep(X, factors, {3: 2})


def test_tucker2_hooi_sweep_updates_both_modes():
    weight = _conv_weight()
    rank_out, rank_in = 32, 16
    _, factors = hosvd(weight, {0: rank_out, 1: rank_in})

    updated = tucker2_hooi_sweep(weight, factors, rank_out, rank_in)

    assert not torch.equal(updated[0], factors[0])
    assert not torch.equal(updated[1], factors[1])


def test_has_converged_requires_keyword_args_for_tolerances():
    with pytest.raises(TypeError):
        has_converged(0.1, 0.2, 1e-8, 1e-5)


def test_core_from_factors_matches_hosvd_core_at_init():
    X = _tensor_3d()
    core_h, factors_h = hosvd(X, RANKS_3D)
    core_from = core_from_factors(X, factors_h)
    assert torch.allclose(core_h, core_from, atol=1e-5)
