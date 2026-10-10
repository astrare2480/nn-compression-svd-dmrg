"""学習可能な ``TTLinear``・``build_tt_linear``・``factorize_named_tt_linear`` の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/00_mlp_ttlinear_logits_equivalence.ipynb
"""

from __future__ import annotations

import copy
import io
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

import nn_compression.compression as compression
import nn_compression.compression.tt_linear as tt_linear_module
import nn_compression.compression.tt_matrix as tt_matrix_module
from nn_compression.compression import (
    TTLinear,
    build_tt_linear,
    dense_to_tt_matrix_cores,
    factorize_named_tt_linear,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)

# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------

# d -> (out_modes, in_modes)
MODES = {
    1: ((6,), (4,)),
    2: ((2, 3), (3, 2)),
    3: ((2, 3, 2), (2, 2, 3)),
}
DTYPES = [torch.float32, torch.float64]


def _tol(dtype: torch.dtype) -> float:
    return 1e-4 if dtype == torch.float32 else 1e-10


def _linear(
    out_modes,
    in_modes,
    *,
    bias: bool,
    dtype: torch.dtype,
    seed: int = 0,
    device=None,
) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(
        math.prod(in_modes), math.prod(out_modes), bias=bias, dtype=dtype, device=device
    )


def _input(linear: nn.Linear, batch: int = 5, seed: int = 100) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        batch,
        linear.in_features,
        dtype=linear.weight.dtype,
        device=linear.weight.device,
    )


def _random_cores(out_modes, in_modes, ranks, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    return [
        torch.randn(ranks[k], m, n, ranks[k + 1], dtype=dtype)
        for k, (m, n) in enumerate(zip(out_modes, in_modes))
    ]


def _all_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


# ---------------------------------------------------------------------------
# TTLinear: forward 値
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_exact_conversion_matches_nn_linear_forward(d, bias, dtype):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=bias, dtype=dtype)
    x = _input(linear)

    layer = build_tt_linear(linear, out_modes, in_modes)

    assert isinstance(layer, TTLinear)
    torch.testing.assert_close(layer(x), linear(x), rtol=_tol(dtype), atol=_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_truncated_conversion_matches_dense_reconstruction_f_linear(d, bias, dtype):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=bias, dtype=dtype, seed=1)
    x = _input(linear)

    layer = build_tt_linear(linear, out_modes, in_modes, max_rank=2)

    cores = list(layer.cores)
    assert all(rank <= 2 for rank in tt_matrix_ranks(cores)[1:-1])
    W = tt_matrix_to_dense(cores, out_modes, in_modes)
    expected = F.linear(x, W, layer.bias)
    torch.testing.assert_close(layer(x), expected, rtol=_tol(dtype), atol=_tol(dtype))


@pytest.mark.parametrize("d", [2, 3])
def test_truncation_actually_changes_the_function(d):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64, seed=2)
    x = _input(linear)

    truncated = build_tt_linear(linear, out_modes, in_modes, max_rank=1)

    assert not torch.allclose(truncated(x), linear(x), rtol=1e-6, atol=1e-6)


def test_forward_does_not_reconstruct_dense_weight(monkeypatch):
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x = _input(linear)
    expected = layer(x)

    def boom(*args, **kwargs):
        raise AssertionError("forward 中に dense weight を復元してはならない")

    monkeypatch.setattr(tt_matrix_module, "tt_matrix_to_dense", boom)
    monkeypatch.setattr(tt_linear_module, "tt_matrix_to_dense", boom, raising=False)
    monkeypatch.setattr(F, "linear", boom)

    torch.testing.assert_close(layer(x), expected, rtol=0.0, atol=0.0)


def test_forward_calls_tt_linear_forward_with_module_state(monkeypatch):
    out_modes, in_modes = MODES[2]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x = _input(linear)

    calls = []
    real = tt_linear_module.tt_linear_forward

    def spy(x_in, cores, out_modes_in, in_modes_in, bias=None):
        calls.append((x_in, cores, out_modes_in, in_modes_in, bias))
        return real(x_in, cores, out_modes_in, in_modes_in, bias)

    monkeypatch.setattr(tt_linear_module, "tt_linear_forward", spy)
    layer(x)

    assert len(calls) == 1
    x_in, cores, out_modes_in, in_modes_in, bias = calls[0]
    assert x_in is x
    assert isinstance(cores, list)
    assert all(a is b for a, b in zip(cores, layer.cores))
    assert out_modes_in == layer.out_modes
    assert in_modes_in == layer.in_modes
    assert bias is layer.bias


# ---------------------------------------------------------------------------
# TTLinear: Parameter 構成・属性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3])
def test_cores_are_registered_as_parameter_list(d):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    layer = build_tt_linear(linear, out_modes, in_modes)

    assert isinstance(layer.cores, nn.ParameterList)
    assert len(layer.cores) == d
    for k, core in enumerate(layer.cores):
        assert isinstance(core, nn.Parameter)
        assert core.is_leaf
        assert core.requires_grad
        assert core.ndim == 4
        assert core.shape[1] == out_modes[k]
        assert core.shape[2] == in_modes[k]


def test_bias_is_parameter_or_none():
    out_modes, in_modes = MODES[2]
    with_bias = build_tt_linear(
        _linear(out_modes, in_modes, bias=True, dtype=torch.float64), out_modes, in_modes
    )
    without_bias = build_tt_linear(
        _linear(out_modes, in_modes, bias=False, dtype=torch.float64), out_modes, in_modes
    )

    assert isinstance(with_bias.bias, nn.Parameter)
    assert with_bias.bias.requires_grad
    assert with_bias.bias.shape == (math.prod(out_modes),)
    assert without_bias.bias is None
    assert "bias" not in dict(without_bias.named_parameters())
    assert "bias" not in without_bias.state_dict()
    assert len(list(without_bias.buffers())) == 0
    assert len(list(with_bias.buffers())) == 0  # biasはbufferではない


@pytest.mark.parametrize("bias", [True, False])
def test_named_parameters_contain_cores_and_bias(bias):
    out_modes, in_modes = MODES[3]
    layer = build_tt_linear(
        _linear(out_modes, in_modes, bias=bias, dtype=torch.float64), out_modes, in_modes
    )

    # PyTorch は自モジュールの Parameter（bias）を子（cores）より先に列挙するため、
    # 順序ではなく集合で比較する。
    names = [name for name, _ in layer.named_parameters()]
    expected = ["cores.0", "cores.1", "cores.2"] + (["bias"] if bias else [])
    assert sorted(names) == sorted(expected)
    assert len(names) == len(set(names))
    assert all(p.requires_grad for p in layer.parameters())


def test_features_and_modes_are_exposed():
    layer = TTLinear(
        _random_cores((2, 3), (3, 2), (1, 4, 1)),
        [2, 3],
        [3, 2],
    )

    assert layer.out_modes == (2, 3)
    assert layer.in_modes == (3, 2)
    assert all(isinstance(mode, int) for mode in layer.out_modes + layer.in_modes)
    assert layer.out_features == 6
    assert layer.in_features == 6
    assert "ranks=(1, 4, 1)" in repr(layer)


# ---------------------------------------------------------------------------
# TTLinear: constructor contract
# ---------------------------------------------------------------------------


def test_constructor_owns_detached_clones_and_does_not_mutate_sources():
    out_modes, in_modes = MODES[2]
    cores = _random_cores(out_modes, in_modes, (1, 3, 1), seed=3)
    bias = torch.randn(math.prod(out_modes), dtype=torch.float64)
    saved_cores = [core.clone() for core in cores]
    saved_bias = bias.clone()

    layer = TTLinear(cores, out_modes, in_modes, bias)

    for source, owned in zip(cores, layer.cores):
        assert owned.data_ptr() != source.data_ptr()
        torch.testing.assert_close(owned, source, rtol=0.0, atol=0.0)
    assert layer.bias.data_ptr() != bias.data_ptr()

    with torch.no_grad():
        for owned in layer.cores:
            owned.add_(1.0)
        layer.bias.add_(1.0)
    for source, saved in zip(cores, saved_cores):
        torch.testing.assert_close(source, saved, rtol=0.0, atol=0.0)
    torch.testing.assert_close(bias, saved_bias, rtol=0.0, atol=0.0)


def test_constructor_detaches_autograd_history_of_sources():
    out_modes, in_modes = MODES[2]
    base = [core.requires_grad_(True) for core in _random_cores(out_modes, in_modes, (1, 3, 1))]
    non_leaf_cores = [core * 1.0 for core in base]
    assert all(core.grad_fn is not None for core in non_leaf_cores)

    layer = TTLinear(non_leaf_cores, out_modes, in_modes)

    for core in layer.cores:
        assert core.is_leaf
        assert core.grad_fn is None
        assert core.requires_grad


def test_constructor_accepts_tuple_of_cores():
    cores = tuple(_random_cores((2, 3), (3, 2), (1, 2, 1)))
    layer = TTLinear(cores, (2, 3), (3, 2))
    assert len(layer.cores) == 2


def test_constructor_rejects_empty_cores():
    with pytest.raises(ValueError):
        TTLinear([], (2,), (2,))


def test_constructor_rejects_non_sequence_cores():
    with pytest.raises(ValueError):
        TTLinear(torch.randn(1, 2, 2, 1, dtype=torch.float64), (2,), (2,))


def test_constructor_rejects_mode_count_mismatch():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(ValueError):
        TTLinear(cores, (2,), (3, 2))


@pytest.mark.parametrize("bad_mode", [0, -1])
def test_constructor_rejects_non_positive_mode(bad_mode):
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(ValueError):
        TTLinear(cores, (bad_mode, 3), (3, 2))


@pytest.mark.parametrize("bad_mode", [True, False, 2.0])
def test_constructor_rejects_bool_and_non_integer_mode(bad_mode):
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(TypeError):
        TTLinear(cores, (bad_mode, 3), (3, 2))


def test_constructor_rejects_physical_shape_mismatch():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 4), (3, 2))


def test_constructor_rejects_bond_mismatch():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    cores[1] = torch.randn(3, 3, 2, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 3), (3, 2))


def test_constructor_rejects_boundary_rank_not_one():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    cores[0] = torch.randn(2, 2, 3, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 3), (3, 2))


def test_constructor_rejects_wrong_core_ndim():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    cores[0] = cores[0].reshape(1, 6, 2)
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 3), (3, 2))


def test_constructor_rejects_dtype_mismatch_between_cores():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    cores[1] = cores[1].to(torch.float32)
    with pytest.raises(TypeError):
        TTLinear(cores, (2, 3), (3, 2))


def test_constructor_rejects_unsupported_dtype():
    cores = [core.to(torch.float16) for core in _random_cores((2, 3), (3, 2), (1, 2, 1))]
    with pytest.raises(TypeError):
        TTLinear(cores, (2, 3), (3, 2))


def test_constructor_rejects_bias_shape_mismatch():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 3), (3, 2), bias=torch.zeros(7, dtype=torch.float64))
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 3), (3, 2), bias=torch.zeros(1, 6, dtype=torch.float64))


def test_constructor_rejects_bias_dtype_mismatch():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(TypeError):
        TTLinear(cores, (2, 3), (3, 2), bias=torch.zeros(6, dtype=torch.float32))


def test_constructor_rejects_bias_that_is_not_a_tensor():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(TypeError):
        TTLinear(cores, (2, 3), (3, 2), bias=[0.0] * 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_constructor_rejects_bias_device_mismatch():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 3), (3, 2), bias=torch.zeros(6, dtype=torch.float64, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_constructor_rejects_device_mismatch_between_cores():
    cores = _random_cores((2, 3), (3, 2), (1, 2, 1))
    cores[1] = cores[1].to("cuda")
    with pytest.raises(ValueError):
        TTLinear(cores, (2, 3), (3, 2))


def test_forward_rejects_invalid_input():
    layer = TTLinear(_random_cores((2, 3), (3, 2), (1, 2, 1)), (2, 3), (3, 2))
    with pytest.raises(ValueError):
        layer(torch.randn(4, 7, dtype=torch.float64))  # 最終次元が不一致
    with pytest.raises(ValueError):
        layer(torch.randn(2, 4, 6, dtype=torch.float64))  # 2階以外
    with pytest.raises(TypeError):
        layer(torch.randn(4, 6, dtype=torch.float32))  # dtype不一致


# ---------------------------------------------------------------------------
# TTLinear: Module としての振る舞い
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_state_dict_roundtrip_reproduces_forward(dtype, bias):
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=bias, dtype=dtype, seed=5)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x = _input(linear)

    state = layer.state_dict()
    expected_keys = ["cores.0", "cores.1", "cores.2"] + (["bias"] if bias else [])
    assert sorted(state.keys()) == sorted(expected_keys)

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded_state = torch.load(buffer, weights_only=True)

    # 構造は同じで値が異なる層へ復元する。
    other_cores = [torch.randn_like(core) for core in layer.cores]
    other_bias = torch.randn_like(layer.bias) if bias else None
    restored = TTLinear(other_cores, out_modes, in_modes, other_bias)
    assert not torch.allclose(restored(x), layer(x), rtol=1e-3, atol=1e-3)

    restored.load_state_dict(loaded_state)

    torch.testing.assert_close(restored(x), layer(x), rtol=0.0, atol=0.0)


def test_load_state_dict_rejects_rank_mismatch():
    out_modes, in_modes = MODES[2]
    small = TTLinear(_random_cores(out_modes, in_modes, (1, 2, 1)), out_modes, in_modes)
    large = TTLinear(_random_cores(out_modes, in_modes, (1, 3, 1)), out_modes, in_modes)
    with pytest.raises(RuntimeError):
        large.load_state_dict(small.state_dict())


def test_to_dtype_moves_cores_and_bias():
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float32, seed=6)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x32 = _input(linear)
    y32 = layer(x32)

    returned = layer.to(torch.float64)

    assert returned is layer
    assert all(core.dtype == torch.float64 for core in layer.cores)
    assert layer.bias.dtype == torch.float64
    assert all(isinstance(core, nn.Parameter) and core.requires_grad for core in layer.cores)
    y64 = layer(x32.to(torch.float64))
    torch.testing.assert_close(y64.to(torch.float32), y32, rtol=1e-4, atol=1e-4)

    layer.float()
    assert all(core.dtype == torch.float32 for core in layer.cores)
    assert layer.bias.dtype == torch.float32


def test_to_device_moves_cores_and_bias_using_meta_device():
    out_modes, in_modes = MODES[2]
    layer = build_tt_linear(
        _linear(out_modes, in_modes, bias=True, dtype=torch.float64), out_modes, in_modes
    )

    layer.to("meta")

    assert all(core.device.type == "meta" for core in layer.cores)
    assert layer.bias.device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
@pytest.mark.parametrize("dtype", DTYPES)
def test_to_cuda_moves_parameters_and_forward_matches_cpu(dtype):
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=True, dtype=dtype, seed=7)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x = _input(linear)
    expected = layer(x)

    layer.to("cuda")

    assert all(core.device.type == "cuda" for core in layer.cores)
    assert layer.bias.device.type == "cuda"
    y_cuda = layer(x.to("cuda"))
    assert y_cuda.device.type == "cuda"
    torch.testing.assert_close(y_cuda.cpu(), expected, rtol=_tol(dtype), atol=_tol(dtype))
    with pytest.raises(ValueError):
        layer(x)  # CPU入力は device 不一致


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_build_tt_linear_from_cuda_linear_keeps_device_and_dtype():
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float32, device="cuda")
    layer = build_tt_linear(linear, out_modes, in_modes, max_rank=3)

    assert all(core.device.type == "cuda" and core.dtype == torch.float32 for core in layer.cores)
    assert layer.bias.device.type == "cuda"
    x = _input(linear)
    assert layer(x).device.type == "cuda"


def test_train_and_eval_behave_as_regular_module():
    out_modes, in_modes = MODES[2]
    layer = build_tt_linear(
        _linear(out_modes, in_modes, bias=True, dtype=torch.float64), out_modes, in_modes
    )
    x = torch.randn(3, 6, dtype=torch.float64)

    assert layer.training
    y_train = layer(x)
    assert layer.eval() is layer
    assert not layer.training
    assert not layer.cores.training
    y_eval = layer(x)
    assert layer.train() is layer
    assert layer.training
    assert layer.cores.training
    layer.train(False)
    assert not layer.training

    torch.testing.assert_close(y_train, y_eval, rtol=0.0, atol=0.0)


def test_deepcopy_gives_independent_parameters():
    out_modes, in_modes = MODES[2]
    layer = build_tt_linear(
        _linear(out_modes, in_modes, bias=True, dtype=torch.float64), out_modes, in_modes
    )
    clone = copy.deepcopy(layer)

    with torch.no_grad():
        clone.cores[0].add_(1.0)

    assert not torch.equal(clone.cores[0], layer.cores[0])
    assert clone.out_modes == layer.out_modes


# ---------------------------------------------------------------------------
# autograd・optimizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_backward_gives_finite_gradients(d, bias, dtype):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=bias, dtype=dtype, seed=8)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x = _input(linear).requires_grad_(True)

    (layer(x) ** 2).sum().backward()

    assert x.grad is not None and _all_finite(x.grad)
    for core in layer.cores:
        assert core.grad is not None and _all_finite(core.grad)
    if bias:
        assert layer.bias.grad is not None and _all_finite(layer.bias.grad)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_gradients_match_dense_reconstruction_path(d, bias):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=bias, dtype=torch.float64, seed=9)
    layer = build_tt_linear(linear, out_modes, in_modes, max_rank=2)
    x_a = _input(linear).requires_grad_(True)
    x_b = x_a.detach().clone().requires_grad_(True)

    cores_b = [core.detach().clone().requires_grad_(True) for core in layer.cores]
    bias_b = layer.bias.detach().clone().requires_grad_(True) if bias else None

    (layer(x_a) ** 2).sum().backward()
    W = tt_matrix_to_dense(cores_b, out_modes, in_modes)
    (F.linear(x_b, W, bias_b) ** 2).sum().backward()

    torch.testing.assert_close(x_a.grad, x_b.grad, rtol=1e-9, atol=1e-9)
    for core, core_b in zip(layer.cores, cores_b):
        torch.testing.assert_close(core.grad, core_b.grad, rtol=1e-9, atol=1e-9)
    if bias:
        torch.testing.assert_close(layer.bias.grad, bias_b.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("bias", [True, False])
def test_gradcheck_small_float64_module(bias):
    out_modes, in_modes = (2, 2), (2, 2)
    cores = _random_cores(out_modes, in_modes, (1, 2, 1), seed=10)
    bias_tensor = torch.randn(4, dtype=torch.float64) if bias else None
    layer = TTLinear(cores, out_modes, in_modes, bias_tensor)

    names = [name for name, _ in layer.named_parameters()]
    params = [param.detach().clone().requires_grad_(True) for _, param in layer.named_parameters()]
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)

    def fn(x_in, *values):
        return functional_call(layer, dict(zip(names, values)), (x_in,))

    assert torch.autograd.gradcheck(fn, (x, *params), atol=1e-6, rtol=1e-4)


def test_optimizer_step_updates_cores_and_bias_and_forward_still_runs():
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64, seed=11)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x = _input(linear)
    target = torch.randn(x.shape[0], linear.out_features, dtype=torch.float64)

    before_cores = [core.detach().clone() for core in layer.cores]
    before_bias = layer.bias.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05)

    optimizer.zero_grad()
    F.mse_loss(layer(x), target).backward()
    optimizer.step()

    assert any(
        not torch.equal(core.detach(), before)
        for core, before in zip(layer.cores, before_cores)
    )
    assert not torch.equal(layer.bias.detach(), before_bias)
    y = layer(x)
    assert y.shape == (x.shape[0], linear.out_features)
    assert _all_finite(y)


def test_fine_tuning_recovers_a_representable_target():
    """表現可能な教師 TT-matrix に摂動を加えた生徒が、学習で教師へ近づく。"""
    out_modes, in_modes = MODES[2]
    teacher_cores = _random_cores(out_modes, in_modes, (1, 3, 1), seed=12)
    teacher_bias = torch.randn(6, dtype=torch.float64)
    teacher = TTLinear(teacher_cores, out_modes, in_modes, teacher_bias)
    x = torch.randn(64, 6, dtype=torch.float64)
    with torch.no_grad():
        target = teacher(x)

    torch.manual_seed(13)
    student = TTLinear(
        [core + 0.3 * torch.randn_like(core) for core in teacher_cores],
        out_modes,
        in_modes,
        teacher_bias + 0.3 * torch.randn_like(teacher_bias),
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    initial_loss = F.mse_loss(student(x), target).item()
    for _ in range(300):
        optimizer.zero_grad()
        F.mse_loss(student(x), target).backward()
        optimizer.step()

    assert F.mse_loss(student(x), target).item() < 0.1 * initial_loss


def test_fine_tuning_from_truncated_conversion_does_not_increase_loss():
    out_modes, in_modes = MODES[2]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64, seed=12)
    x = _input(linear, batch=32)
    with torch.no_grad():
        target = linear(x)

    layer = build_tt_linear(linear, out_modes, in_modes, max_rank=1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-2)
    initial_loss = F.mse_loss(layer(x), target).item()
    for _ in range(100):
        optimizer.zero_grad()
        F.mse_loss(layer(x), target).backward()
        optimizer.step()

    assert F.mse_loss(layer(x), target).item() < initial_loss


def test_frozen_parameters_are_not_updated():
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64, seed=13)
    layer = build_tt_linear(linear, out_modes, in_modes)
    x = _input(linear)
    target = torch.randn(x.shape[0], linear.out_features, dtype=torch.float64)

    layer.cores[0].requires_grad_(False)
    layer.bias.requires_grad_(False)
    frozen_core = layer.cores[0].detach().clone()
    frozen_bias = layer.bias.detach().clone()
    free_before = [core.detach().clone() for core in layer.cores[1:]]
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05, weight_decay=0.1)

    optimizer.zero_grad()
    F.mse_loss(layer(x), target).backward()
    optimizer.step()

    assert layer.cores[0].grad is None
    assert layer.bias.grad is None
    torch.testing.assert_close(layer.cores[0].detach(), frozen_core, rtol=0.0, atol=0.0)
    torch.testing.assert_close(layer.bias.detach(), frozen_bias, rtol=0.0, atol=0.0)
    assert all(
        not torch.equal(core.detach(), before)
        for core, before in zip(layer.cores[1:], free_before)
    )


# ---------------------------------------------------------------------------
# build_tt_linear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_rank", [1, 2, 3])
@pytest.mark.parametrize("d", [2, 3])
def test_build_tt_linear_positive_max_rank_bounds_bond_ranks(d, max_rank):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64, seed=14)

    layer = build_tt_linear(linear, out_modes, in_modes, max_rank=max_rank)

    ranks = tt_matrix_ranks(list(layer.cores))
    assert ranks[0] == 1 and ranks[-1] == 1
    assert all(rank <= max_rank for rank in ranks[1:-1])


def test_build_tt_linear_max_rank_none_keeps_full_rank():
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64, seed=15)

    exact = build_tt_linear(linear, out_modes, in_modes, max_rank=None)
    capped = build_tt_linear(linear, out_modes, in_modes, max_rank=1)

    assert max(tt_matrix_ranks(list(exact.cores))) > max(tt_matrix_ranks(list(capped.cores)))


@pytest.mark.parametrize("max_rank", [None, 1, 5])
def test_build_tt_linear_one_site(max_rank):
    out_modes, in_modes = MODES[1]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64, seed=16)
    x = _input(linear)

    layer = build_tt_linear(linear, out_modes, in_modes, max_rank=max_rank)

    assert len(layer.cores) == 1
    assert tuple(layer.cores[0].shape) == (1, 6, 4, 1)
    torch.testing.assert_close(layer(x), linear(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("bias", [True, False])
def test_build_tt_linear_does_not_share_storage_or_mutate_linear(d, bias):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=bias, dtype=torch.float64, seed=17)
    saved_weight = linear.weight.detach().clone()
    saved_bias = linear.bias.detach().clone() if bias else None

    layer = build_tt_linear(linear, out_modes, in_modes)

    weight_storage = linear.weight.untyped_storage().data_ptr()
    for core in layer.cores:
        assert core.untyped_storage().data_ptr() != weight_storage
    if bias:
        assert layer.bias.data_ptr() != linear.bias.data_ptr()
    torch.testing.assert_close(linear.weight.detach(), saved_weight, rtol=0.0, atol=0.0)
    if bias:
        torch.testing.assert_close(linear.bias.detach(), saved_bias, rtol=0.0, atol=0.0)
    assert linear.weight.requires_grad  # 元の状態も変えない

    # 双方向に独立: どちらを書き換えても他方へ伝わらない。
    x = _input(linear)
    expected = layer(x).detach().clone()
    with torch.no_grad():
        linear.weight.add_(1.0)
        if bias:
            linear.bias.add_(1.0)
    torch.testing.assert_close(layer(x), expected, rtol=0.0, atol=0.0)
    with torch.no_grad():
        for core in layer.cores:
            core.add_(1.0)
    torch.testing.assert_close(
        linear.weight.detach(), saved_weight + 1.0, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_build_tt_linear_keeps_dtype_device_and_bias_presence(dtype):
    out_modes, in_modes = MODES[2]
    with_bias = build_tt_linear(
        _linear(out_modes, in_modes, bias=True, dtype=dtype), out_modes, in_modes
    )
    without_bias = build_tt_linear(
        _linear(out_modes, in_modes, bias=False, dtype=dtype), out_modes, in_modes
    )

    for layer in (with_bias, without_bias):
        assert all(core.dtype == dtype and core.device.type == "cpu" for core in layer.cores)
    assert with_bias.bias.dtype == dtype
    assert without_bias.bias is None


def test_build_tt_linear_inherits_requires_grad_state():
    out_modes, in_modes = MODES[2]

    default = build_tt_linear(
        _linear(out_modes, in_modes, bias=True, dtype=torch.float64), out_modes, in_modes
    )
    assert all(core.requires_grad for core in default.cores)
    assert default.bias.requires_grad

    frozen_weight = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    frozen_weight.weight.requires_grad_(False)
    layer = build_tt_linear(frozen_weight, out_modes, in_modes)
    assert not any(core.requires_grad for core in layer.cores)
    assert layer.bias.requires_grad  # biasは独立に引き継ぐ

    frozen_bias = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    frozen_bias.bias.requires_grad_(False)
    layer = build_tt_linear(frozen_bias, out_modes, in_modes)
    assert all(core.requires_grad for core in layer.cores)
    assert not layer.bias.requires_grad

    fully_frozen = _linear(out_modes, in_modes, bias=True, dtype=torch.float64).requires_grad_(False)
    layer = build_tt_linear(fully_frozen, out_modes, in_modes)
    assert not any(parameter.requires_grad for parameter in layer.parameters())


def test_build_tt_linear_mirrors_training_flag():
    out_modes, in_modes = MODES[2]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    assert build_tt_linear(linear, out_modes, in_modes).training

    linear.eval()
    layer = build_tt_linear(linear, out_modes, in_modes)
    assert not layer.training
    assert not layer.cores.training


def test_build_tt_linear_matches_dense_to_tt_matrix_cores():
    out_modes, in_modes = MODES[3]
    linear = _linear(out_modes, in_modes, bias=False, dtype=torch.float64, seed=18)

    layer = build_tt_linear(linear, out_modes, in_modes, max_rank=2)
    expected = dense_to_tt_matrix_cores(
        linear.weight.detach(), out_modes, in_modes, max_rank=2
    )

    assert len(layer.cores) == len(expected)
    for core, reference in zip(layer.cores, expected):
        torch.testing.assert_close(core.detach(), reference, rtol=0.0, atol=0.0)


def test_build_tt_linear_rejects_non_linear():
    with pytest.raises(TypeError):
        build_tt_linear(nn.ReLU(), (2, 3), (3, 2))
    with pytest.raises(TypeError):
        build_tt_linear(torch.randn(6, 6, dtype=torch.float64), (2, 3), (3, 2))


def test_build_tt_linear_rejects_shape_mismatch():
    linear = _linear((2, 3), (3, 2), bias=True, dtype=torch.float64)
    with pytest.raises(ValueError):
        build_tt_linear(linear, (2, 4), (3, 2))
    with pytest.raises(ValueError):
        build_tt_linear(linear, (2, 3), (3, 3))


def test_build_tt_linear_rejects_mode_count_mismatch_and_empty_modes():
    linear = _linear((2, 3), (3, 2), bias=True, dtype=torch.float64)
    with pytest.raises(ValueError):
        build_tt_linear(linear, (2, 3), (6,))
    with pytest.raises(ValueError):
        build_tt_linear(linear, (), ())


@pytest.mark.parametrize("bad_mode", [True, 2.0, "2"])
def test_build_tt_linear_rejects_invalid_mode_type(bad_mode):
    linear = _linear((2, 3), (3, 2), bias=True, dtype=torch.float64)
    with pytest.raises(TypeError):
        build_tt_linear(linear, (bad_mode, 3), (3, 2))


@pytest.mark.parametrize("bad_mode", [0, -2])
def test_build_tt_linear_rejects_non_positive_mode(bad_mode):
    linear = _linear((2, 3), (3, 2), bias=True, dtype=torch.float64)
    with pytest.raises(ValueError):
        build_tt_linear(linear, (bad_mode, 3), (3, 2))


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("max_rank", [0, -1])
def test_build_tt_linear_rejects_non_positive_max_rank(d, max_rank):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    with pytest.raises(ValueError):
        build_tt_linear(linear, out_modes, in_modes, max_rank=max_rank)


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("max_rank", [True, False, 1.5, "2"])
def test_build_tt_linear_rejects_invalid_max_rank_type(d, max_rank):
    out_modes, in_modes = MODES[d]
    linear = _linear(out_modes, in_modes, bias=True, dtype=torch.float64)
    with pytest.raises(TypeError):
        build_tt_linear(linear, out_modes, in_modes, max_rank=max_rank)


def test_build_tt_linear_rejects_unsupported_dtype():
    linear = _linear((2, 3), (3, 2), bias=True, dtype=torch.float32).half()
    with pytest.raises(TypeError):
        build_tt_linear(linear, (2, 3), (3, 2))


# ---------------------------------------------------------------------------
# factorize_named_tt_linear
# ---------------------------------------------------------------------------


class _Net(nn.Module):
    """ネストした path・非 Linear module・buffer を含む小さな model。"""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 8))
        self.norm = nn.BatchNorm1d(8)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(8, 4)
        self.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.act(self.norm(self.encoder(x))))


# layer_name -> (out_modes, in_modes)
NET_TARGETS = {
    "encoder.0": ((3, 4), (2, 3)),  # 6 -> 12
    "encoder.2": ((2, 4), (3, 4)),  # 12 -> 8
    "classifier": ((2, 2), (2, 4)),  # 8 -> 4
}


def _make_net(dtype=torch.float64, seed=0) -> _Net:
    torch.manual_seed(seed)
    net = _Net(dtype)
    # BatchNorm の統計も非自明な値にする。
    with torch.no_grad():
        net.norm.running_mean.normal_()
        net.norm.running_var.uniform_(0.5, 1.5)
    return net


def _net_input(dtype=torch.float64, batch=5) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randn(batch, 6, dtype=dtype)


@pytest.mark.parametrize("layer_name", list(NET_TARGETS))
def test_named_replacement_replaces_only_target_in_copy(layer_name):
    out_modes, in_modes = NET_TARGETS[layer_name]
    model = _make_net().eval()

    compressed = factorize_named_tt_linear(model, layer_name, out_modes, in_modes)

    assert compressed is not model
    assert isinstance(compressed.get_submodule(layer_name), TTLinear)
    assert isinstance(model.get_submodule(layer_name), nn.Linear)

    original_names = {
        name: type(module)
        for name, module in model.named_modules()
        if name != layer_name
    }
    copied_names = {
        name: type(module)
        for name, module in compressed.named_modules()
        if name != layer_name
        and not name.startswith(layer_name + ".")  # TTLinear 内部の ParameterList
    }
    assert copied_names == original_names


@pytest.mark.parametrize("layer_name", list(NET_TARGETS))
def test_named_replacement_does_not_change_baseline_model(layer_name):
    out_modes, in_modes = NET_TARGETS[layer_name]
    model = _make_net().train()
    saved_state = {key: value.clone() for key, value in model.state_dict().items()}
    saved_flags = {name: module.training for name, module in model.named_modules()}
    saved_ids = {name: id(module) for name, module in model.named_modules()}
    saved_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}

    compressed = factorize_named_tt_linear(model, layer_name, out_modes, in_modes, max_rank=2)

    assert list(model.state_dict().keys()) == list(saved_state.keys())
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved_state[key]), key
    assert {name: module.training for name, module in model.named_modules()} == saved_flags
    assert {name: id(module) for name, module in model.named_modules()} == saved_ids
    assert {n: p.requires_grad for n, p in model.named_parameters()} == saved_requires_grad

    # 更新しても baseline の Parameter は動かない（deepcopy の独立性）。
    optimizer = torch.optim.SGD(compressed.parameters(), lr=0.1)
    compressed.train()
    compressed(_net_input()).sum().backward()
    optimizer.step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved_state[key]), key


@pytest.mark.parametrize("layer_name", list(NET_TARGETS))
def test_named_replacement_keeps_non_target_parameters_independent_copies(layer_name):
    out_modes, in_modes = NET_TARGETS[layer_name]
    model = _make_net()

    compressed = factorize_named_tt_linear(model, layer_name, out_modes, in_modes)

    original = dict(model.named_parameters())
    for name, parameter in compressed.named_parameters():
        if name.startswith(layer_name + "."):
            continue
        assert name in original
        assert type(parameter) is type(original[name])
        assert parameter.data_ptr() != original[name].data_ptr()
        torch.testing.assert_close(parameter, original[name], rtol=0.0, atol=0.0)
    for name, buffer in model.named_buffers():
        assert torch.equal(dict(compressed.named_buffers())[name], buffer)


@pytest.mark.parametrize("layer_name", list(NET_TARGETS))
def test_named_replacement_exact_keeps_model_logits(layer_name):
    out_modes, in_modes = NET_TARGETS[layer_name]
    model = _make_net().eval()
    x = _net_input()

    compressed = factorize_named_tt_linear(model, layer_name, out_modes, in_modes)
    compressed.eval()

    torch.testing.assert_close(compressed(x), model(x), rtol=1e-10, atol=1e-10)


def test_named_replacement_can_be_applied_repeatedly():
    model = _make_net().eval()
    x = _net_input()

    compressed = model
    for layer_name, (out_modes, in_modes) in NET_TARGETS.items():
        compressed = factorize_named_tt_linear(compressed, layer_name, out_modes, in_modes)

    assert all(isinstance(compressed.get_submodule(n), TTLinear) for n in NET_TARGETS)
    torch.testing.assert_close(compressed(x), model(x), rtol=1e-9, atol=1e-9)
    assert all(isinstance(model.get_submodule(n), nn.Linear) for n in NET_TARGETS)


def test_named_replacement_with_max_rank_bounds_rank_and_matches_dense_path():
    model = _make_net().eval()
    out_modes, in_modes = NET_TARGETS["encoder.2"]

    compressed = factorize_named_tt_linear(model, "encoder.2", out_modes, in_modes, max_rank=2)

    layer = compressed.encoder[2]
    assert all(rank <= 2 for rank in tt_matrix_ranks(list(layer.cores))[1:-1])
    x = torch.randn(4, 12, dtype=torch.float64)
    W = tt_matrix_to_dense(list(layer.cores), out_modes, in_modes)
    torch.testing.assert_close(layer(x), F.linear(x, W, layer.bias), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", DTYPES)
def test_named_replacement_keeps_dtype_and_training_state(dtype):
    model = _make_net(dtype).train()
    model.classifier.eval()
    flags = {name: module.training for name, module in model.named_modules()}
    out_modes, in_modes = NET_TARGETS["encoder.0"]

    compressed = factorize_named_tt_linear(model, "encoder.0", out_modes, in_modes)

    assert {p.dtype for p in compressed.parameters()} == {dtype}
    assert all(p.device.type == "cpu" for p in compressed.parameters())
    for name, module in compressed.named_modules():
        if name == "encoder.0" or name.startswith("encoder.0."):
            continue
        assert module.training == flags[name], name
    # 置換された層の training は元の Linear に揃う。
    assert compressed.encoder[0].training == model.encoder[0].training

    model.eval()
    compressed_eval = factorize_named_tt_linear(model, "encoder.0", out_modes, in_modes)
    assert not compressed_eval.encoder[0].training
    assert not compressed_eval.training


def test_named_replacement_inherits_requires_grad_of_target_layer():
    model = _make_net()
    model.encoder[0].weight.requires_grad_(False)
    out_modes, in_modes = NET_TARGETS["encoder.0"]

    compressed = factorize_named_tt_linear(model, "encoder.0", out_modes, in_modes)

    assert not any(core.requires_grad for core in compressed.encoder[0].cores)
    assert compressed.encoder[0].bias.requires_grad
    assert compressed.classifier.weight.requires_grad


def test_named_replacement_rejects_missing_path():
    model = _make_net()
    with pytest.raises(AttributeError):
        factorize_named_tt_linear(model, "encoder.9", (3, 4), (2, 3))
    with pytest.raises(AttributeError):
        factorize_named_tt_linear(model, "missing", (3, 4), (2, 3))


@pytest.mark.parametrize("layer_name", ["encoder.1", "encoder", "norm", "act"])
def test_named_replacement_rejects_non_linear_target(layer_name):
    model = _make_net()
    saved = {key: value.clone() for key, value in model.state_dict().items()}

    with pytest.raises(TypeError):
        factorize_named_tt_linear(model, layer_name, (3, 4), (2, 3))

    for key, value in model.state_dict().items():
        assert torch.equal(value, saved[key])


def test_named_replacement_rejects_already_replaced_layer():
    model = _make_net()
    compressed = factorize_named_tt_linear(model, "encoder.0", *NET_TARGETS["encoder.0"])
    with pytest.raises(TypeError):
        factorize_named_tt_linear(compressed, "encoder.0", *NET_TARGETS["encoder.0"])


def test_named_replacement_rejects_invalid_layer_name():
    model = _make_net()
    with pytest.raises(ValueError):
        factorize_named_tt_linear(model, "", (3, 4), (2, 3))
    with pytest.raises(TypeError):
        factorize_named_tt_linear(model, None, (3, 4), (2, 3))
    with pytest.raises(TypeError):
        factorize_named_tt_linear(model, 0, (3, 4), (2, 3))


@pytest.mark.parametrize(
    ("out_modes", "in_modes", "max_rank", "error"),
    [
        ((3, 5), (2, 3), None, ValueError),  # shape 不一致
        ((True, 12), (2, 3), None, TypeError),  # bool mode
        ((0, 4), (2, 3), None, ValueError),  # 非正 mode
        ((3, 4), (2, 3), 0, ValueError),  # 非正 rank
        ((3, 4), (2, 3), True, TypeError),  # bool rank
    ],
)
def test_named_replacement_invalid_arguments_leave_model_untouched(
    out_modes, in_modes, max_rank, error
):
    model = _make_net()
    saved = {key: value.clone() for key, value in model.state_dict().items()}

    with pytest.raises(error):
        factorize_named_tt_linear(model, "encoder.0", out_modes, in_modes, max_rank=max_rank)

    assert isinstance(model.encoder[0], nn.Linear)
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_named_replacement_keeps_cuda_device_and_logits():
    model = _make_net(torch.float32).to("cuda").eval()
    x = _net_input(torch.float32).to("cuda")
    out_modes, in_modes = NET_TARGETS["encoder.0"]

    compressed = factorize_named_tt_linear(model, "encoder.0", out_modes, in_modes)

    assert {p.device.type for p in compressed.parameters()} == {"cuda"}
    torch.testing.assert_close(compressed.eval()(x), model(x), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["TTLinear", "build_tt_linear", "factorize_named_tt_linear", "ordered_factorizations"],
)
def test_public_api_is_exported(name):
    assert name in compression.__all__
    assert hasattr(compression, name)
