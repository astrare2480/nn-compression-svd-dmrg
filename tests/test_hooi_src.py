"""HOOI 共通化の回帰テスト。

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
LOW_RANKS_3D = {0: 3, 1: 2, 2: 2}


def _tensor_3d(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*SHAPE_3D)


def _conv_weight(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(64, 32, 3, 3)


def test_hooi_sweep_does_not_mutate_input_factors():
    X = _tensor_3d()
    ranks = LOW_RANKS_3D
    _, factors = hosvd(X, ranks)
    original = {mode: U.clone() for mode, U in factors.items()}

    updated = hooi_sweep(X, factors, ranks)

    for mode in factors:
        assert torch.equal(factors[mode], original[mode])
    assert updated is not factors
    for mode in ranks:
        assert updated[mode].shape == (SHAPE_3D[mode], ranks[mode])


def test_hooi_works_for_three_mode_tensor():
    X = _tensor_3d()
    core, factors, history = hooi(X, LOW_RANKS_3D, max_iter=5)

    assert tuple(core.shape) == (3, 2, 2)
    assert factors[0].shape == (6, 3)
    assert factors[1].shape == (5, 2)
    assert factors[2].shape == (4, 2)
    assert len(history) >= 2
    assert reconstruct_tucker(core, factors).shape == X.shape


def test_partial_hooi_works_for_conv_weight():
    weight = _conv_weight()
    rank_out, rank_in = 32, 16
    ranks = {0: rank_out, 1: rank_in}

    core, factors, history = hooi(weight, ranks, max_iter=5)

    assert tuple(core.shape) == (rank_out, rank_in, 3, 3)
    assert factors[0].shape == (64, rank_out)
    assert factors[1].shape == (32, rank_in)
    assert len(history) >= 2
    assert reconstruct_tucker(core, factors).shape == weight.shape


def test_tucker2_hooi_core_and_factor_shapes():
    weight = _conv_weight()
    rank_out, rank_in = 32, 16

    core, factors, _ = tucker2_hooi(weight, rank_out, rank_in, max_iter=3)

    assert tuple(core.shape) == (rank_out, rank_in, 3, 3)
    assert factors[0].shape == (64, rank_out)
    assert factors[1].shape == (32, rank_in)


def test_tucker2_hooi_sweep_updates_both_modes():
    weight = _conv_weight()
    rank_out, rank_in = 32, 16
    _, factors = hosvd(weight, {0: rank_out, 1: rank_in})

    updated = tucker2_hooi_sweep(weight, factors, rank_out, rank_in)

    assert not torch.equal(updated[0], factors[0])
    assert not torch.equal(updated[1], factors[1])


def test_hooi_error_does_not_worsen_from_hosvd_initialization():
    X = _tensor_3d()
    _, _, history = hooi(X, LOW_RANKS_3D, max_iter=10)

    assert history[0] >= history[-1]


def test_hooi_history_starts_with_hosvd_error():
    X = _tensor_3d()
    ranks = LOW_RANKS_3D
    core_h, factors_h = hosvd(X, ranks)
    hosvd_error = float(
        relative_frobenius_error(X, reconstruct_tucker(core_h, factors_h))
    )

    _, _, history = hooi(X, ranks, max_iter=3)

    assert history[0] == pytest.approx(hosvd_error, rel=0, abs=1e-6)


def test_hooi_history_is_nonincreasing():
    X = _tensor_3d()
    _, _, history = hooi(X, LOW_RANKS_3D, max_iter=10)

    for prev, curr in zip(history, history[1:]):
        assert curr <= prev + 1e-8


def test_has_converged_keyword_only_args():
    with pytest.raises(TypeError):
        has_converged(0.1, 0.2, 1e-8, 1e-5)


def test_hooi_rejects_invalid_mode_in_sweep():
    X = _tensor_3d()
    _, factors = hosvd(X, LOW_RANKS_3D)
    with pytest.raises(ValueError):
        hooi_sweep(X, factors, {3: 2})


def test_hooi_rejects_missing_factor():
    X = _tensor_3d()
    _, factors = hosvd(X, {0: 3, 1: 2})
    with pytest.raises(ValueError, match="factors に mode=2"):
        hooi_sweep(X, factors, {0: 3, 1: 2, 2: 2})


def test_hooi_rejects_invalid_mode():
    X = _tensor_3d()
    with pytest.raises(ValueError, match="mode="):
        hooi(X, {3: 2}, max_iter=1)


def test_core_from_factors_matches_hosvd_core_at_init():
    X = _tensor_3d()
    ranks = LOW_RANKS_3D
    core_h, factors_h = hosvd(X, ranks)
    core_from = core_from_factors(X, factors_h)
    assert torch.allclose(core_h, core_from, atol=1e-5)
