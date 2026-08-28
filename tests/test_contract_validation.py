"""Codex レビュー指摘に対応する validation / contract テスト。"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from nn_compression.compression import (
    build_tucker2_conv,
    build_tucker2_conv_from_components,
    compression_factor,
    has_converged,
    hooi,
    hooi_sweep,
    hosvd,
    parameter_ratio,
    reconstruct_tucker,
    tucker2_decompose_conv_weight,
    tucker2_effective_weight,
    tucker2_hooi,
    tucker_parameter_count,
)
from nn_compression.compression.linear_svd import factorize_linear_layer
from nn_compression.compression.svd import coerce_integer_scalar, coerce_rank, truncated_svd
from nn_compression.compression.tucker_validation import validate_max_iter
from nn_compression.metrics import (
    agreement,
    benchmark_inference,
    benchmark_inference_print,
    collect_compression_metrics,
    compressed_conv2d_macs,
    compressed_linear_macs,
    estimate_conv2d_macs,
    logits_rmse,
    relative_frobenius_error,
)
from nn_compression.tensor import fold, mode_dot, unfold
from nn_compression.training import evaluate, train_one_epoch


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


@pytest.mark.parametrize("ranks", [None, [], (0, 3)])
def test_tucker_ranks_rejects_non_mapping(ranks):
    with pytest.raises(TypeError, match="Mapping"):
        tucker_parameter_count((6, 5, 4), ranks)


@pytest.mark.parametrize("rank", [True, False])
def test_coerce_rank_rejects_bool(rank):
    with pytest.raises(TypeError, match="bool"):
        coerce_rank(rank, 5)


np = pytest.importorskip("numpy")


@pytest.mark.parametrize("rank", [np.bool_(True), np.bool_(False)])
def test_coerce_rank_rejects_numpy_bool(rank):
    with pytest.raises(TypeError, match="bool"):
        coerce_rank(rank, 5)


@pytest.mark.parametrize("rank", [torch.tensor(True), torch.tensor(False)])
def test_coerce_rank_rejects_torch_bool_tensor(rank):
    with pytest.raises(TypeError, match="Tensor scalar"):
        coerce_rank(rank, 5)


def test_coerce_rank_accepts_numpy_int64():
    assert coerce_rank(np.int64(2), 5) == 2


@pytest.mark.parametrize("max_iter", [np.bool_(True), np.bool_(False), torch.tensor(True)])
def test_validate_max_iter_rejects_boolean_scalars(max_iter):
    with pytest.raises(TypeError):
        validate_max_iter(max_iter)


def test_validate_max_iter_accepts_numpy_int64():
    assert validate_max_iter(np.int64(3)) == 3


@pytest.mark.parametrize("rank", [True, False])
def test_hosvd_rejects_bool_rank(rank):
    X = torch.randn(6, 5, 4)
    with pytest.raises(TypeError, match="bool"):
        hosvd(X, {0: rank})


@pytest.mark.parametrize("rank", [True, False])
def test_tucker_parameter_count_rejects_bool_rank(rank):
    with pytest.raises(TypeError, match="bool"):
        tucker_parameter_count((6, 5, 4), {0: rank})


@pytest.mark.parametrize("rank", [True, False])
def test_hooi_rejects_bool_rank(rank):
    X = torch.randn(6, 5, 4)
    with pytest.raises(TypeError, match="bool"):
        hooi(X, {0: rank, 1: 2, 2: 2}, max_iter=1)


@pytest.mark.parametrize("rank", [True, False])
def test_tucker2_decompose_rejects_bool_rank(rank):
    weight = torch.randn(8, 4, 3, 3)
    with pytest.raises(TypeError, match="bool"):
        tucker2_decompose_conv_weight(weight, rank_out=rank, rank_in=2)


def test_tucker_parameter_count_rejects_unfolding_impossible_rank():
    with pytest.raises(ValueError):
        tucker_parameter_count((100, 2), {0: 50})


def test_hosvd_rejects_unfolding_impossible_rank():
    X = torch.randn(100, 2)
    with pytest.raises(ValueError):
        hosvd(X, {0: 50})


def test_tucker2_decompose_rejects_unfolding_impossible_rank_out():
    """Tucker-2入口を通過後、generic Tucker validationで拒否される rank。"""
    weight = torch.randn(100, 2, 3, 3)
    with pytest.raises(ValueError):
        tucker2_decompose_conv_weight(weight, rank_out=50, rank_in=2)


@pytest.mark.parametrize("max_iter", [0, 1])
def test_hooi_rejects_impossible_rank_independent_of_max_iter(max_iter):
    X = torch.randn(10, 10)
    with pytest.raises(ValueError, match="projected unfolding"):
        hooi(X, {0: 5, 1: 1}, max_iter=max_iter)


def test_hooi_rejects_bool_mode_in_ranks():
    X = torch.randn(6, 5, 4)
    with pytest.raises(TypeError, match="bool"):
        hooi(X, {True: 2, 1: 2, 2: 2}, max_iter=1)


def test_hooi_sweep_rejects_bool_mode_in_factors():
    X = torch.randn(6, 5, 4)
    _, factors = hosvd(X, {0: 3, 1: 2})
    factors = {True: factors[0], 1: factors[1]}
    with pytest.raises(TypeError, match="factor mode"):
        hooi_sweep(X, factors, {0: 3, 1: 2})


@pytest.mark.parametrize("abs_tol", [-1.0, float("nan"), float("inf"), True])
def test_has_converged_rejects_invalid_abs_tol(abs_tol):
    with pytest.raises((TypeError, ValueError)):
        has_converged(0.1, 0.2, abs_tol=abs_tol, rel_tol=1e-5)


def test_has_converged_allows_zero_tolerances():
    assert has_converged(0.0, 0.0, abs_tol=0.0, rel_tol=0.0) is True


# --- Tensor ndim contract ---


def test_unfold_rejects_0d_and_1d_tensors():
    with pytest.raises(ValueError, match="2次元以上"):
        unfold(torch.tensor(1.0), 0)
    with pytest.raises(ValueError, match="2次元以上"):
        unfold(torch.randn(5), 0)


def test_unfold_rejects_1d_tensor():
    with pytest.raises(ValueError, match="2次元以上"):
        unfold(torch.randn(5), 0)


def test_fold_rejects_1d_target_shape():
    with pytest.raises(ValueError, match="2次元以上"):
        fold(torch.randn(5, 1), 0, (5,))


def test_fold_unfold_roundtrip_maintained():
    X = torch.randn(6, 5, 4)
    for mode in range(X.ndim):
        assert torch.equal(fold(unfold(X, mode), mode, X.shape), X)


@pytest.mark.parametrize("factory", [unfold, mode_dot])
def test_tensor_ops_reject_zero_size_dimension(factory):
    X = torch.randn(0, 5)
    if factory is unfold:
        with pytest.raises(ValueError, match="正の整数"):
            factory(X, 0)
    else:
        matrix = torch.randn(2, 0)
        with pytest.raises(ValueError, match="正の整数"):
            factory(X, matrix, mode=0)


def test_fold_rejects_zero_size_shape():
    with pytest.raises(ValueError, match="正の整数"):
        fold(torch.randn(2, 3), 0, (0, 5))


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


# --- 実数 dtype contract（複素数は明示的に未対応） ---


def test_hosvd_rejects_complex_dtype():
    X = torch.randn(3, 4, 2, dtype=torch.complex128)
    with pytest.raises(TypeError, match="複素数"):
        hosvd(X, {0: 2, 1: 2, 2: 2})


def test_hooi_rejects_complex_dtype():
    X = torch.randn(3, 4, 2, dtype=torch.complex128)
    with pytest.raises(TypeError, match="複素数"):
        hooi(X, {0: 2, 1: 2, 2: 2}, max_iter=1)


def test_hooi_sweep_rejects_complex_dtype():
    X_real = torch.randn(3, 4, 2)
    _, factors_real = hosvd(X_real, {0: 2, 1: 2})
    X = X_real.to(torch.complex128)
    factors = {m: U.to(torch.complex128) for m, U in factors_real.items()}
    with pytest.raises(TypeError, match="複素数"):
        hooi_sweep(X, factors, {0: 2, 1: 2})


def test_hosvd_full_rank_reconstruction_unaffected_by_complex_guard():
    """複素数を拒否する guard 追加後も、実数 full-rank 再構成精度は変わらない。"""
    torch.manual_seed(0)
    X = torch.randn(3, 4, 2, dtype=torch.float64)
    core, factors = hosvd(X, {0: 3, 1: 4, 2: 2})
    X_hat = reconstruct_tucker(core, factors)
    relative_error = (
        torch.linalg.vector_norm(X - X_hat) / torch.linalg.vector_norm(X)
    ).item()
    assert relative_error < 1e-10


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


# --- compressed MACs rank contract (SVD 側と統一) ---


@pytest.mark.parametrize("rank", [True, False, 0, -1, 5])
def test_compressed_linear_macs_rejects_invalid_rank(rank):
    # in_features=4, out_features=3 -> max_rank = min(4, 3) = 3
    with pytest.raises((TypeError, ValueError)):
        compressed_linear_macs(4, 3, rank)


def test_compressed_linear_macs_accepts_boundary_rank():
    assert compressed_linear_macs(4, 3, 3) == 4 * 3 + 3 * 3


@pytest.mark.parametrize("rank", [True, False, 0, -1, 99])
def test_compressed_conv2d_macs_rejects_invalid_rank(rank):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    # out_ch=5, in_ch*kH*kW=27 -> max_rank = min(5, 27) = 5
    with pytest.raises((TypeError, ValueError)):
        compressed_conv2d_macs(conv, rank=rank, out_h=8, out_w=8)


def test_compressed_conv2d_macs_accepts_boundary_rank():
    conv = nn.Conv2d(3, 5, kernel_size=3)
    macs = compressed_conv2d_macs(conv, rank=5, out_h=8, out_w=8)
    assert macs == 8 * 8 * 5 * 3 * 9 + 8 * 8 * 5 * 5


@pytest.mark.parametrize("rank", [True, 0, -1, 99])
def test_estimate_conv2d_macs_rejects_invalid_rank(rank):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    with pytest.raises((TypeError, ValueError)):
        estimate_conv2d_macs(conv, rank=rank, out_hw=(8, 8), verbose=False)


def test_baseline_linear_conv_macs_unaffected_by_rank_validation():
    # rank validation は compressed 側のみ。baseline はrankを取らない。
    from nn_compression.metrics import conv2d_macs, linear_macs

    layer = nn.Linear(4, 3)
    assert linear_macs(layer) == 12
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


def test_benchmark_inference_print_restores_training_mode_when_train():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    baseline.train(True)
    compressed.train(True)
    batch = torch.randn(2, 4)
    loader = _metric_loader()
    benchmark_inference_print(
        baseline,
        compressed,
        loader,
        torch.device("cpu"),
        verbose=False,
        warmup=0,
        repeats=1,
        input_batch=batch,
    )
    assert baseline.training is True
    assert compressed.training is True


def test_benchmark_inference_print_restores_training_mode_when_eval():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    baseline.eval()
    compressed.eval()
    batch = torch.randn(2, 4)
    loader = _metric_loader()
    benchmark_inference_print(
        baseline,
        compressed,
        loader,
        torch.device("cpu"),
        verbose=False,
        warmup=0,
        repeats=1,
        input_batch=batch,
    )
    assert baseline.training is False
    assert compressed.training is False


def test_benchmark_inference_print_restores_training_mode_on_exception():
    from unittest.mock import patch

    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    baseline.train(True)
    compressed.train(True)
    batch = torch.randn(2, 4)
    loader = _metric_loader()
    with patch(
        "nn_compression.metrics.model_comparison.benchmark_inference",
        side_effect=RuntimeError("benchmark failed"),
    ):
        with pytest.raises(RuntimeError, match="benchmark failed"):
            benchmark_inference_print(
                baseline,
                compressed,
                loader,
                torch.device("cpu"),
                verbose=False,
                warmup=0,
                repeats=1,
                input_batch=batch,
            )
    assert baseline.training is True
    assert compressed.training is True


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


class _FeatureIterable(IterableDataset):
    def __init__(self, count: int) -> None:
        self.count = count

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randn(4), torch.randint(0, 3, ())


def test_agreement_accepts_iterable_dataset_without_len():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    loader = DataLoader(_FeatureIterable(8), batch_size=4)
    value = agreement(baseline, compressed, loader, torch.device("cpu"))
    assert 0.0 <= value <= 1.0


def test_agreement_rejects_empty_iterable_dataset():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    loader = DataLoader(_FeatureIterable(0), batch_size=4)
    with pytest.raises(ValueError, match="空"):
        agreement(baseline, compressed, loader, torch.device("cpu"))


def test_evaluate_restores_training_mode_after_success():
    model = nn.Linear(4, 3)
    model.train(True)
    loader = _metric_loader()
    evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert model.training is True


def test_evaluate_restores_training_mode_when_already_eval():
    model = nn.Linear(4, 3)
    model.eval()
    loader = _metric_loader()
    evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert model.training is False


def test_evaluate_restores_training_mode_on_forward_error():
    class BrokenModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("forward failed")

    model = BrokenModel()
    model.train(True)
    loader = _metric_loader()
    with pytest.raises(RuntimeError, match="forward failed"):
        evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert model.training is True


def test_evaluate_rejects_empty_loader():
    model = nn.Linear(4, 3)
    empty = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    with pytest.raises(ValueError, match="空"):
        evaluate(model, empty, nn.CrossEntropyLoss(), torch.device("cpu"))


# --- submodule 個別の training 状態復元 ---


def _mixed_state_model() -> nn.Module:
    """root=train() だが、内部の BatchNorm だけ eval() にしたモデル。"""
    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.BatchNorm1d(4),
        nn.Linear(4, 3),
    )
    model.train()
    model[1].eval()  # frozen BatchNorm を想定
    return model


def _capture_training_states(model: nn.Module) -> list[bool]:
    return [module.training for module in model.modules()]


def test_evaluate_restores_all_submodule_training_states_after_success():
    model = _mixed_state_model()
    before = _capture_training_states(model)
    loader = _metric_loader()
    evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert _capture_training_states(model) == before


def test_evaluate_restores_all_submodule_training_states_on_exception():
    model = _mixed_state_model()
    before = _capture_training_states(model)

    def _broken_forward(x):
        raise RuntimeError("forward failed")

    model.forward = _broken_forward
    loader = _metric_loader()
    with pytest.raises(RuntimeError, match="forward failed"):
        evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert _capture_training_states(model) == before


def test_evaluate_submodule_training_states_unchanged_when_root_eval():
    model = _mixed_state_model()
    model.eval()
    model[0].train()  # root=eval だが一部 submodule だけ train() にする
    before = _capture_training_states(model)
    loader = _metric_loader()
    evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert _capture_training_states(model) == before


def test_agreement_restores_all_submodule_training_states():
    baseline = _mixed_state_model()
    compressed = _mixed_state_model()
    before_baseline = _capture_training_states(baseline)
    before_compressed = _capture_training_states(compressed)
    loader = _metric_loader()
    agreement(baseline, compressed, loader, torch.device("cpu"))
    assert _capture_training_states(baseline) == before_baseline
    assert _capture_training_states(compressed) == before_compressed


def test_benchmark_inference_restores_all_submodule_training_states():
    model = _mixed_state_model()
    before = _capture_training_states(model)
    batch = torch.randn(4, 4)
    benchmark_inference(
        model,
        device=torch.device("cpu"),
        warmup=0,
        repeats=1,
        input_batch=batch,
    )
    assert _capture_training_states(model) == before


def test_collect_compression_metrics_restores_all_submodule_training_states():
    baseline = _mixed_state_model()
    compressed = _mixed_state_model()
    before_baseline = _capture_training_states(baseline)
    before_compressed = _capture_training_states(compressed)
    loader = _metric_loader()
    collect_compression_metrics(
        baseline,
        compressed,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        baseline_acc=0.5,
        baseline_time_s=0.001,
        warmup=0,
        repeats=1,
    )
    assert _capture_training_states(baseline) == before_baseline
    assert _capture_training_states(compressed) == before_compressed


# --- train_one_epoch の empty loader contract ---


def test_train_one_epoch_rejects_empty_dataloader():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    empty = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    with pytest.raises(ValueError, match="空"):
        train_one_epoch(
            model, empty, nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
        )


def test_train_one_epoch_rejects_empty_iterable_without_len():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(_FeatureIterable(0), batch_size=4)
    with pytest.raises(ValueError, match="空"):
        train_one_epoch(
            model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
        )


def test_train_one_epoch_normal_behavior_unaffected():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _metric_loader()
    avg_loss, accuracy = train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )
    assert avg_loss >= 0.0
    assert 0.0 <= accuracy <= 1.0


# --- benchmark_inference の warmup / repeats contract ---


@pytest.mark.parametrize(
    "warmup,repeats",
    [
        (False, 1),
        (-1, 1),
        (1.5, 1),
        (1, False),
        (1, 0),
        (1, -1),
        (1, 1.5),
    ],
)
def test_benchmark_inference_rejects_invalid_warmup_repeats(warmup, repeats):
    model = nn.Linear(4, 3)
    batch = torch.randn(2, 4)
    with pytest.raises((TypeError, ValueError)):
        benchmark_inference(
            model,
            device=torch.device("cpu"),
            warmup=warmup,
            repeats=repeats,
            input_batch=batch,
        )


def test_benchmark_inference_accepts_boundary_warmup_and_repeats():
    model = nn.Linear(4, 3)
    batch = torch.randn(2, 4)
    time_s = benchmark_inference(
        model,
        device=torch.device("cpu"),
        warmup=0,
        repeats=1,
        input_batch=batch,
    )
    assert time_s >= 0.0


def test_collect_compression_metrics_preserves_training_modes():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    baseline.train(True)
    compressed.train(True)
    loader = _metric_loader()
    collect_compression_metrics(
        baseline,
        compressed,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        baseline_acc=0.5,
        baseline_time_s=0.001,
        warmup=0,
        repeats=1,
    )
    assert baseline.training is True
    assert compressed.training is True


def test_logits_rmse_restores_training_mode_on_forward_error():
    class BrokenBaseline(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("logits failed")

    baseline = BrokenBaseline()
    compressed = nn.Linear(4, 3)
    baseline.train(True)
    compressed.train(True)
    loader = _metric_loader()
    with pytest.raises(RuntimeError, match="logits failed"):
        logits_rmse(baseline, compressed, loader, torch.device("cpu"))
    assert baseline.training is True
    assert compressed.training is True


def test_logits_rmse_accepts_iterable_dataset_without_len():
    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    loader = DataLoader(_FeatureIterable(8), batch_size=4)
    value = logits_rmse(baseline, compressed, loader, torch.device("cpu"))
    assert value >= 0.0


def test_collect_compression_metrics_restores_training_on_evaluate_error():
    from unittest.mock import patch

    baseline = nn.Linear(4, 3)
    compressed = nn.Linear(4, 3)
    baseline.train(True)
    compressed.train(True)
    loader = _metric_loader()
    with patch(
        "nn_compression.metrics.model_comparison.evaluate",
        side_effect=RuntimeError("evaluate failed"),
    ):
        with pytest.raises(RuntimeError, match="evaluate failed"):
            collect_compression_metrics(
                baseline,
                compressed,
                loader,
                nn.CrossEntropyLoss(),
                torch.device("cpu"),
                baseline_acc=0.5,
                baseline_time_s=0.001,
                warmup=0,
                repeats=1,
            )
    assert baseline.training is True
    assert compressed.training is True


def test_hosvd_preserves_requires_grad():
    weight = torch.randn(8, 4, 3, 3, requires_grad=True)
    core, u_out, u_in = tucker2_decompose_conv_weight(weight, 4, 2)
    assert core.requires_grad
    assert u_out.requires_grad
    assert u_in.requires_grad
    (core.sum() + u_out.sum() + u_in.sum()).backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


def test_hooi_preserves_requires_grad():
    X = torch.randn(6, 5, 4, requires_grad=True)
    core, factors, _ = hooi(X, {0: 3, 1: 2, 2: 2}, max_iter=1)
    assert core.requires_grad
    assert all(U.requires_grad for U in factors.values())
    loss = core.sum() + sum(U.sum() for U in factors.values())
    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_tucker2_hooi_preserves_requires_grad():
    weight = torch.randn(8, 4, 3, 3, requires_grad=True)
    core, factors, _ = tucker2_hooi(weight, rank_out=4, rank_in=2, max_iter=1)
    assert core.requires_grad
    assert factors[0].requires_grad
    assert factors[1].requires_grad
    loss = core.sum() + factors[0].sum() + factors[1].sum()
    loss.backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


def test_build_tucker2_conv_parameters_are_leaf_and_not_shared():
    conv = _conv2d()
    original_ptr = conv.weight.data_ptr()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    params = [p for layer in seq for p in layer.parameters()]
    assert params
    assert all(p.is_leaf for p in params)
    assert all(p.data_ptr() != original_ptr for p in params)


def test_builder_seq_passes_effective_weight_validation():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    effective = tucker2_effective_weight(seq)
    assert effective.shape == conv.weight.shape


def test_builder_seq_forward_matches_effective_weight_conv_float64():
    conv = _conv2d(dtype=torch.float64)
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    effective = tucker2_effective_weight(seq)
    equivalent = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=conv.bias is not None,
        dtype=torch.float64,
    )
    with torch.no_grad():
        equivalent.weight.copy_(effective)
        if conv.bias is not None:
            equivalent.bias.copy_(conv.bias)
    x = torch.randn(2, conv.in_channels, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        y_seq = seq(x)
        y_equiv = equivalent(x)
    torch.testing.assert_close(y_seq, y_equiv, atol=1e-12, rtol=1e-12)


def test_tucker2_effective_weight_rejects_non_three_layers():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    with pytest.raises(ValueError, match="3層"):
        tucker2_effective_weight(nn.Sequential(seq[0], seq[1]))


def test_tucker2_effective_weight_rejects_non_conv_layer():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    bad = nn.Sequential(nn.Linear(4, 4), seq[1], seq[2])
    with pytest.raises(TypeError, match="Conv2d"):
        tucker2_effective_weight(bad)


def test_tucker2_effective_weight_rejects_output_side_stride():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    seq[2].stride = (2, 2)
    with pytest.raises(ValueError, match="stride"):
        tucker2_effective_weight(seq)


def test_tucker2_effective_weight_rejects_output_side_groups():
    conv = _conv2d()
    seq = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    seq[2].groups = 2
    with pytest.raises(ValueError, match="groups"):
        tucker2_effective_weight(seq)


def test_build_tucker2_from_components_rejects_1d_u_out():
    conv = _conv2d()
    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), 32, 16
    )
    with pytest.raises(ValueError, match="2次元"):
        build_tucker2_conv_from_components(conv, core, u_out[:, 0], u_in)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が利用できません")
def test_build_tucker2_from_components_rejects_device_mismatch():
    conv = _conv2d(device=torch.device("cuda"))
    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), 32, 16
    )
    with pytest.raises(ValueError, match="device"):
        build_tucker2_conv_from_components(conv, core.cpu(), u_out, u_in)


@pytest.mark.parametrize("rank", [True, False])
def test_linear_svd_rejects_bool_rank(rank):
    layer = nn.Linear(4, 3)
    with pytest.raises(TypeError, match="bool"):
        factorize_linear_layer(layer, rank)
