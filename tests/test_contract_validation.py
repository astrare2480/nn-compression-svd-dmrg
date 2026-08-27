"""Codex レビュー指摘に対応する validation / contract テスト。"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import (
    build_tucker2_conv,
    build_tucker2_conv_from_components,
    compression_factor,
    hooi,
    hooi_sweep,
    hosvd,
    parameter_ratio,
    tucker2_decompose_conv_weight,
    tucker2_effective_weight,
    tucker_parameter_count,
)
from nn_compression.compression.svd import truncated_svd
from nn_compression.metrics import (
    agreement,
    benchmark_inference,
    compressed_conv2d_macs,
    estimate_conv2d_macs,
    logits_rmse,
)
from nn_compression.tensor import fold, mode_dot, unfold


def _conv2d(**kwargs) -> nn.Conv2d:
    defaults = dict(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    defaults.update(kwargs)
    conv = nn.Conv2d(**defaults)
    torch.manual_seed(0)
    with torch.no_grad():
        conv.weight.normal_()
        if conv.bias is not None:
            conv.bias.normal_()
    return conv


def _tiny_loader(n: int = 8):
    images = torch.randn(n, 1, 8, 8)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


# --- Tucker parameter count / compression ---


@pytest.mark.parametrize("mode", ["0", True, 1.5])
def test_tucker_parameter_count_rejects_invalid_mode(mode):
    with pytest.raises((TypeError, ValueError)):
        tucker_parameter_count((6, 5, 4), {mode: 2})


def test_tucker_parameter_count_rejects_out_of_range_mode():
    with pytest.raises(ValueError, match="mode"):
        tucker_parameter_count((6, 5, 4), {3: 2})


@pytest.mark.parametrize("rank", [0, -1, 7])
def test_tucker_parameter_count_rejects_invalid_rank(rank):
    with pytest.raises(ValueError):
        tucker_parameter_count((6, 5, 4), {0: rank})


def test_tucker_parameter_count_allows_empty_ranks_as_identity():
    shape = (6, 5, 4)
    assert tucker_parameter_count(shape, {}) == 6 * 5 * 4
    assert parameter_ratio(shape, {}) == pytest.approx(1.0)
    assert compression_factor(shape, {}) == pytest.approx(1.0)


def test_tucker_metrics_reject_zero_dimension_shape():
    with pytest.raises(ValueError):
        tucker_parameter_count((0, 5, 4), {0: 1})


# --- Tensor ndim contract ---


def test_unfold_rejects_0d_and_1d_tensors():
    with pytest.raises(ValueError, match="2次元以上"):
        unfold(torch.tensor(1.0), 0)
    with pytest.raises(ValueError, match="2次元以上"):
        unfold(torch.randn(5), 0)


def test_unfold_rejects_1d_tensor():
    with pytest.raises(ValueError, match="2次元以上"):
        unfold(torch.randn(5), 0)


def test_fold_rejects_invalid_shape_for_1d_target():
    with pytest.raises(ValueError):
        fold(torch.randn(2, 3), 0, (5,))


# --- fold validation ---


def test_fold_rejects_non_2d_unfolded():
    X = torch.randn(6, 5, 4)
    with pytest.raises(ValueError, match="2次元"):
        fold(X, 0, X.shape)


def test_fold_rejects_transposed_unfolded():
    X = torch.randn(6, 5, 4)
    unfolded = unfold(X, 0)
    transposed = unfolded.T
    with pytest.raises(ValueError):
        fold(transposed, 0, X.shape)


def test_fold_rejects_shape_mismatch():
    X = torch.randn(6, 5, 4)
    unfolded = unfold(X, 0)
    with pytest.raises(ValueError):
        fold(unfolded, 0, (7, 5, 4))


def test_fold_rejects_bool_mode():
    X = torch.randn(6, 5, 4)
    with pytest.raises(TypeError):
        fold(unfold(X, 0), True, X.shape)


# --- HOOI factor / rank contract ---


def test_hooi_rejects_extra_factor():
    X = torch.randn(6, 5, 4)
    _, factors = hosvd(X, {0: 3, 1: 2})
    factors[2] = torch.randn(4, 2)
    with pytest.raises(ValueError, match="ranks に無い"):
        hooi_sweep(X, factors, {0: 3, 1: 2})


def test_hooi_rejects_missing_factor():
    X = torch.randn(6, 5, 4)
    _, factors = hosvd(X, {0: 3, 1: 2})
    del factors[1]
    with pytest.raises(ValueError, match="mode=1"):
        hooi_sweep(X, factors, {0: 3, 1: 2})


def test_hooi_rejects_impossible_partial_rank():
    X = torch.randn(10, 10)
    _, factors = hosvd(X, {0: 5, 1: 1})
    with pytest.raises(ValueError, match="projected unfolding"):
        hooi_sweep(X, factors, {0: 5, 1: 1})


def test_hooi_rejects_dtype_mismatch():
    X = torch.randn(6, 5, 4, dtype=torch.float64)
    _, factors = hosvd(X, {0: 3, 1: 2})
    factors_f32 = {m: U.float() for m, U in factors.items()}
    with pytest.raises(ValueError, match="dtype"):
        hooi_sweep(X, factors_f32, {0: 3, 1: 2})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が利用できません")
def test_hooi_rejects_device_mismatch():
    X = torch.randn(6, 5, 4, device="cuda")
    _, factors = hosvd(X, {0: 3, 1: 2})
    factors_cpu = {m: U.cpu() for m, U in factors.items()}
    with pytest.raises(ValueError, match="device"):
        hooi_sweep(X, factors_cpu, {0: 3, 1: 2})


@pytest.mark.parametrize("abs_tol", [-1.0, float("nan"), float("inf")])
def test_hooi_rejects_invalid_abs_tol(abs_tol):
    X = torch.randn(6, 5, 4)
    with pytest.raises(ValueError):
        hooi(X, {0: 3, 1: 2, 2: 2}, abs_tol=abs_tol)


@pytest.mark.parametrize("rel_tol", [-0.1, float("nan"), float("inf")])
def test_hooi_rejects_invalid_rel_tol(rel_tol):
    X = torch.randn(6, 5, 4)
    with pytest.raises(ValueError):
        hooi(X, {0: 3, 1: 2, 2: 2}, rel_tol=rel_tol)


@pytest.mark.parametrize("max_iter", [-1, 1.5, True])
def test_hooi_rejects_invalid_max_iter(max_iter):
    X = torch.randn(6, 5, 4)
    with pytest.raises((TypeError, ValueError)):
        hooi(X, {0: 3, 1: 2, 2: 2}, max_iter=max_iter)


def test_hooi_rejects_zero_tensor():
    X = torch.zeros(6, 5, 4)
    with pytest.raises(ValueError, match="ゼロテンソル"):
        hooi(X, {0: 3, 1: 2, 2: 2})


def test_hooi_rejects_empty_ranks():
    X = torch.randn(6, 5, 4)
    with pytest.raises(ValueError, match="空"):
        hooi(X, {}, max_iter=1)


def test_hosvd_allows_empty_ranks():
    X = torch.randn(6, 5, 4)
    core, factors = hosvd(X, {})
    assert torch.equal(core, X)
    assert factors == {}


def test_hooi_sweep_uses_gauss_seidel_update():
    """mode 0 更新後の factor を mode 1 更新に使う（Jacobi 型と区別できる）。"""
    torch.manual_seed(42)
    X = torch.randn(10, 10)
    ranks = {0: 3, 1: 3}
    u0, _ = torch.linalg.qr(torch.randn(10, 3))
    u1, _ = torch.linalg.qr(torch.randn(10, 3))
    factors = {0: u0, 1: u1}

    updated = {mode: U.clone() for mode, U in factors.items()}

    projected_0 = X
    for other_mode in ranks:
        if other_mode == 0:
            continue
        projected_0 = mode_dot(projected_0, updated[other_mode].T, other_mode)
    updated[0], _, _ = truncated_svd(unfold(projected_0, 0), ranks[0])

    projected_1_gs = X
    for other_mode in ranks:
        if other_mode == 1:
            continue
        projected_1_gs = mode_dot(
            projected_1_gs, updated[other_mode].T, other_mode
        )
    u1_gs, _, _ = truncated_svd(unfold(projected_1_gs, 1), ranks[1])

    projected_1_jacobi = mode_dot(X, factors[0].T, 0)
    u1_jacobi, _, _ = truncated_svd(unfold(projected_1_jacobi, 1), ranks[1])

    sweep_result = hooi_sweep(X, factors, ranks)

    assert torch.allclose(sweep_result[0], updated[0], atol=1e-5)
    assert torch.allclose(sweep_result[1], u1_gs, atol=1e-5)
    assert not torch.allclose(sweep_result[1], u1_jacobi, atol=1e-3)


# --- Tucker-2 validation ---


def _metric_loader(n: int = 8):
    features = torch.randn(n, 4)
    labels = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(features, labels), batch_size=4)


def test_tucker2_effective_weight_rejects_invalid_input_stride():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    seq[0].stride = (2, 2)
    with pytest.raises(ValueError, match="stride"):
        tucker2_effective_weight(seq)


def test_tucker2_effective_weight_rejects_invalid_input_padding():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    seq[0].padding = (1, 1)
    with pytest.raises(ValueError, match="padding"):
        tucker2_effective_weight(seq)


def test_tucker2_effective_weight_rejects_non_1x1_input_kernel():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    bad_input = nn.Conv2d(
        seq[0].in_channels,
        seq[0].out_channels,
        kernel_size=3,
        bias=False,
    )
    seq[0] = bad_input
    with pytest.raises(ValueError, match="1x1"):
        tucker2_effective_weight(seq)


def test_tucker2_effective_weight_rejects_input_bias():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    seq[0].bias = nn.Parameter(torch.zeros(seq[0].out_channels))
    with pytest.raises(ValueError, match="input projection"):
        tucker2_effective_weight(seq)


def test_tucker2_effective_weight_rejects_core_bias():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    seq[1].bias = nn.Parameter(torch.zeros(seq[1].out_channels))
    with pytest.raises(ValueError, match="core layer"):
        tucker2_effective_weight(seq)


def test_tucker2_effective_weight_rejects_broken_channel_connection():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    broken_output = nn.Conv2d(
        seq[2].in_channels + 1,
        seq[2].out_channels,
        kernel_size=1,
        bias=seq[2].bias is not None,
    )
    seq[2] = broken_output
    with pytest.raises(ValueError, match="channel"):
        tucker2_effective_weight(seq)


def test_build_tucker2_from_components_rejects_3d_core():
    conv = _conv2d()
    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), 32, 16
    )
    with pytest.raises(ValueError, match="4階"):
        build_tucker2_conv_from_components(conv, core[0], u_out, u_in)


def test_build_tucker2_from_components_rejects_dtype_mismatch():
    conv = _conv2d(dtype=torch.float64)
    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), 32, 16
    )
    with pytest.raises(ValueError, match="dtype"):
        build_tucker2_conv_from_components(conv, core.float(), u_out, u_in)


# --- grouped Conv compressed MACs ---


def test_compressed_conv2d_macs_rejects_grouped_conv():
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2)
    with pytest.raises(ValueError, match="groups=1"):
        compressed_conv2d_macs(conv, rank=2, out_h=8, out_w=8)


def test_estimate_conv2d_macs_rejects_grouped_conv():
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2)
    with pytest.raises(ValueError, match="groups=1"):
        estimate_conv2d_macs(conv, rank=2, out_hw=(8, 8), verbose=False)


def test_baseline_conv2d_macs_allows_grouped_conv():
    from nn_compression.metrics import conv2d_macs

    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2)
    assert conv2d_macs(conv, 8, 8) > 0


# --- metrics train/eval state ---


def test_agreement_restores_training_mode():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    baseline.train(True)
    compressed.train(True)
    loader = _metric_loader()
    agreement(baseline, compressed, loader, torch.device("cpu"))
    assert baseline.training is True
    assert compressed.training is True


def test_logits_rmse_restores_training_mode():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    baseline.train(True)
    compressed.train(True)
    loader = _metric_loader()
    logits_rmse(baseline, compressed, loader, torch.device("cpu"))
    assert baseline.training is True
    assert compressed.training is True


def test_benchmark_inference_restores_training_mode():
    model = nn.Linear(4, 3)
    model.train(True)
    batch = torch.randn(2, 4)
    benchmark_inference(
        model,
        device=torch.device("cpu"),
        warmup=0,
        repeats=1,
        input_batch=batch,
    )
    assert model.training is True


def test_agreement_rejects_empty_loader():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    empty = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    with pytest.raises(ValueError, match="空"):
        agreement(baseline, compressed, empty, torch.device("cpu"))


def test_logits_rmse_rejects_empty_loader():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    empty = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    with pytest.raises(ValueError, match="空"):
        logits_rmse(baseline, compressed, empty, torch.device("cpu"))
