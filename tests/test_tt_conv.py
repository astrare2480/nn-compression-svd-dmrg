"""``TTConv2d``・``tt_conv2d_forward``・``build_tt_conv2d`` など TT-Conv API の回帰テスト。

参照元 Notebook:
    notebooks/30_tt_mps/20_fashion_mnist_cnn/00_ttconv_representation_and_dense_kernel_reconstruction.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/01_ttconv_direct_forward_and_backward_equivalence.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/02_pretrained_cnn_conv2_exact_ttconv_equivalence.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/03_pretrained_cnn_conv2_tt_rank_sweep.ipynb
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import nn_compression.compression.tt_conv as tt_conv_module
import nn_compression.compression.tt_matrix as tt_matrix_module
from nn_compression.compression import (
    TTConv2d,
    build_tt_conv2d,
    dense_to_tt_matrix_cores,
    factorize_named_tt_conv2d,
    sweep_tt_conv2d_weight_ranks,
    sweep_tt_matrix_weight_ranks,
    tt_conv2d_forward,
    tt_conv2d_to_dense_weight,
    tt_matrix_num_parameters,
    tt_matrix_ranks,
)

DTYPES = [torch.float32, torch.float64]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません"),
    ),
]

# in_channels=4, out_channels=8, kernel=(3, 2): patch 幅は 4*3*2 = 24。
IN_CHANNELS = 4
OUT_CHANNELS = 8
KERNEL = (3, 2)
MODES = {
    1: ((8,), (24,)),
    2: ((2, 4), (6, 4)),
    3: ((2, 2, 2), (3, 2, 4)),
}

# 設定を変えた Conv2d 引数（kernel は固定。stride / padding / dilation を変える）。
CONV_SETTINGS = [
    pytest.param({}, id="default"),
    pytest.param({"stride": 2}, id="stride2"),
    pytest.param({"stride": (2, 1)}, id="stride_2_1"),
    pytest.param({"padding": 1}, id="padding1"),
    pytest.param({"padding": (2, 0)}, id="padding_2_0"),
    pytest.param({"dilation": 2, "padding": 2}, id="dilation2"),
    pytest.param({"dilation": (1, 3), "stride": (1, 2), "padding": (1, 2)}, id="mixed"),
]


def _tol(dtype: torch.dtype) -> dict:
    return {"rtol": 1e-4, "atol": 1e-4} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-10}


def _conv(*, bias=True, dtype=torch.float64, device="cpu", seed=0, **kwargs) -> nn.Conv2d:
    torch.manual_seed(seed)
    return nn.Conv2d(
        IN_CHANNELS, OUT_CHANNELS, KERNEL, bias=bias, dtype=dtype, device=device, **kwargs
    )


def _x(conv: nn.Conv2d, batch=2, size=(9, 8), seed=100) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        batch,
        conv.in_channels,
        *size,
        dtype=conv.weight.dtype,
        device=conv.weight.device,
    )


def _layer(d=3, *, bias=True, dtype=torch.float64, device="cpu", max_rank=None, **kwargs):
    conv = _conv(bias=bias, dtype=dtype, device=device, **kwargs)
    out_modes, in_modes = MODES[d]
    return conv, build_tt_conv2d(conv, out_modes, in_modes, max_rank=max_rank)


def _manual_layer(cores, out_modes, in_modes, **kwargs):
    defaults = {
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "kernel_size": KERNEL,
    }
    defaults.update(kwargs)
    return TTConv2d(cores, out_modes, in_modes, **defaults)


def _cores(d=3, *, dtype=torch.float64, device="cpu", max_rank=None, seed=0):
    conv = _conv(dtype=dtype, device=device, seed=seed)
    out_modes, in_modes = MODES[d]
    cores = dense_to_tt_matrix_cores(
        conv.weight.detach().reshape(OUT_CHANNELS, -1), out_modes, in_modes, max_rank=max_rank
    )
    return conv, cores, out_modes, in_modes


# ---------------------------------------------------------------------------
# forward の数値等価
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("settings", CONV_SETTINGS)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_exact_tt_conv_matches_nn_conv2d(d, settings, dtype, device):
    conv, layer = _layer(d, dtype=dtype, device=device, **settings)
    x = _x(conv)

    torch.testing.assert_close(layer(x), conv(x), **_tol(dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("settings", CONV_SETTINGS)
@pytest.mark.parametrize("max_rank", [1, 2])
def test_truncated_tt_conv_matches_dense_kernel_conv(max_rank, settings, dtype, device):
    conv, layer = _layer(3, dtype=dtype, device=device, max_rank=max_rank, **settings)
    x = _x(conv)

    kernel = tt_conv2d_to_dense_weight(
        list(layer.cores), layer.out_modes, layer.in_modes,
        in_channels=IN_CHANNELS, kernel_size=KERNEL,
    )
    expected = F.conv2d(
        x, kernel, layer.bias,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
    )
    torch.testing.assert_close(layer(x), expected, **_tol(dtype))
    # 打ち切りは dense Conv から実際にずれる（比較が自明でないことの確認）。
    assert float((layer(x) - conv(x)).abs().max().detach()) > 1e-3


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("size", [(3, 2), (5, 7), (9, 8), (16, 11)])
def test_forward_matches_for_batches_and_spatial_sizes(batch, size):
    conv, layer = _layer(3, padding=1)
    x = _x(conv, batch=batch, size=size)

    out = layer(x)
    assert out.shape == conv(x).shape
    torch.testing.assert_close(out, conv(x), **_tol(torch.float64))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_non_square_kernel_and_asymmetric_channels(d):
    conv = nn.Conv2d(4, 8, (1, 5), padding=(0, 2), dtype=torch.float64)
    out_modes = MODES[d][0]
    in_modes = {1: (20,), 2: (4, 5), 3: (2, 2, 5)}[d]
    layer = build_tt_conv2d(conv, out_modes, in_modes)
    x = _x(conv, size=(6, 7))

    torch.testing.assert_close(layer(x), conv(x), **_tol(torch.float64))


@pytest.mark.parametrize("d", [1, 3])
def test_no_bias_layer_matches_conv_and_has_no_bias(d):
    conv, layer = _layer(d, bias=False)
    x = _x(conv)

    assert layer.bias is None
    assert "bias" not in dict(layer.named_parameters())
    torch.testing.assert_close(layer(x), conv(x), **_tol(torch.float64))


def test_forward_output_is_contiguous_like_nn_conv2d():
    conv, layer = _layer(3, padding=1)
    out = layer(_x(conv))

    assert out.is_contiguous()
    assert out.view(out.shape[0], -1).shape[0] == out.shape[0]


def test_output_size_formula_matches_nn_conv2d_for_many_settings():
    for stride in (1, 2, 3):
        for padding in (0, 1, 2):
            for dilation in (1, 2):
                conv, layer = _layer(
                    3, stride=stride, padding=padding, dilation=dilation
                )
                x = _x(conv, size=(11, 10))
                assert layer(x).shape == conv(x).shape


# ---------------------------------------------------------------------------
# 勾配
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_gradients_match_dense_reconstruction_path(d, dtype):
    conv, cores, out_modes, in_modes = _cores(d, dtype=dtype, max_rank=2)
    x = _x(conv, size=(7, 6))
    bias = conv.bias.detach().clone()

    def grads(use_direct: bool):
        core_params = [c.detach().clone().requires_grad_(True) for c in cores]
        x_leaf = x.detach().clone().requires_grad_(True)
        bias_leaf = bias.clone().requires_grad_(True)
        if use_direct:
            out = tt_conv2d_forward(
                x_leaf, core_params, out_modes, in_modes, bias_leaf,
                kernel_size=KERNEL, stride=1, padding=1,
            )
        else:
            kernel = tt_conv2d_to_dense_weight(
                core_params, out_modes, in_modes, in_channels=IN_CHANNELS, kernel_size=KERNEL
            )
            out = F.conv2d(x_leaf, kernel, bias_leaf, padding=1)
        target = torch.linspace(-1, 1, out.numel(), dtype=dtype).reshape(out.shape)
        F.mse_loss(out, target).backward()
        return x_leaf.grad, [c.grad for c in core_params], bias_leaf.grad

    direct = grads(True)
    reference = grads(False)

    torch.testing.assert_close(direct[0], reference[0], **_tol(dtype))
    for direct_core, reference_core in zip(direct[1], reference[1]):
        assert direct_core is not None and torch.isfinite(direct_core).all()
        torch.testing.assert_close(direct_core, reference_core, **_tol(dtype))
    torch.testing.assert_close(direct[2], reference[2], **_tol(dtype))


def test_layer_backward_populates_every_core_and_bias_gradient():
    conv, layer = _layer(3, max_rank=3, padding=1)
    layer(_x(conv)).square().mean().backward()

    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.shape == parameter.shape
    for core in layer.cores:
        assert float(torch.linalg.vector_norm(core.grad)) > 0.0


def test_forward_keeps_autograd_graph_from_input_and_cores():
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv).requires_grad_(True)
    core_params = [c.clone().requires_grad_(True) for c in cores]

    out = tt_conv2d_forward(x, core_params, out_modes, in_modes, kernel_size=KERNEL)

    assert out.requires_grad
    assert out.grad_fn is not None


# ---------------------------------------------------------------------------
# forward は dense kernel を作らない・既存 API を使う
# ---------------------------------------------------------------------------


def test_forward_does_not_reconstruct_dense_kernel(monkeypatch):
    conv, layer = _layer(3, padding=1)
    x = _x(conv)

    def boom(*args, **kwargs):
        raise AssertionError("dense kernel を作ってはいけません。")

    monkeypatch.setattr(tt_matrix_module, "tt_matrix_to_dense", boom)
    monkeypatch.setattr(tt_conv_module, "tt_matrix_to_dense", boom)
    monkeypatch.setattr(F, "conv2d", boom)

    layer(x)


def test_forward_delegates_to_tt_linear_forward(monkeypatch):
    conv, layer = _layer(3, padding=1)
    x = _x(conv)
    calls = []
    real = tt_conv_module.tt_linear_forward

    def spy(patch_matrix, cores, out_modes, in_modes, bias=None):
        calls.append((tuple(patch_matrix.shape), len(cores), out_modes, in_modes, bias))
        return real(patch_matrix, cores, out_modes, in_modes, bias)

    monkeypatch.setattr(tt_conv_module, "tt_linear_forward", spy)
    out = layer(x)

    assert len(calls) == 1
    patch_shape, num_cores, out_modes, in_modes, bias = calls[0]
    assert patch_shape == (x.shape[0] * out.shape[2] * out.shape[3], 24)
    assert num_cores == 3
    assert out_modes == layer.out_modes and in_modes == layer.in_modes
    assert bias is layer.bias


def test_module_forward_calls_tt_conv2d_forward(monkeypatch):
    conv, layer = _layer(3, stride=2, padding=1, dilation=1)
    x = _x(conv)
    recorded = {}
    real = tt_conv_module.tt_conv2d_forward

    def spy(x_arg, cores, out_modes, in_modes, bias=None, **kwargs):
        recorded.update(kwargs)
        return real(x_arg, cores, out_modes, in_modes, bias, **kwargs)

    monkeypatch.setattr(tt_conv_module, "tt_conv2d_forward", spy)
    layer(x)

    assert recorded == {
        "kernel_size": KERNEL,
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
    }


# ---------------------------------------------------------------------------
# 非破壊
# ---------------------------------------------------------------------------


def test_forward_does_not_mutate_input_cores_or_bias():
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv)
    bias = conv.bias.detach().clone()
    saved_x, saved_bias = x.clone(), bias.clone()
    saved_cores = [c.clone() for c in cores]

    tt_conv2d_forward(x, cores, out_modes, in_modes, bias, kernel_size=KERNEL, padding=1)

    assert torch.equal(x, saved_x)
    assert torch.equal(bias, saved_bias)
    for core, saved in zip(cores, saved_cores):
        assert torch.equal(core, saved)


# ---------------------------------------------------------------------------
# forward の入力検証
# ---------------------------------------------------------------------------


def test_forward_rejects_bad_x():
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv)

    with pytest.raises(TypeError, match="x"):
        tt_conv2d_forward([[1.0]], cores, out_modes, in_modes, kernel_size=KERNEL)
    with pytest.raises(ValueError, match="4階"):
        tt_conv2d_forward(x[0], cores, out_modes, in_modes, kernel_size=KERNEL)
    with pytest.raises(ValueError, match="4階"):
        tt_conv2d_forward(x.reshape(2, 4, -1), cores, out_modes, in_modes, kernel_size=KERNEL)


def test_forward_rejects_channel_mismatch():
    conv, cores, out_modes, in_modes = _cores(3)
    x = torch.randn(2, IN_CHANNELS + 1, 9, 8, dtype=torch.float64)

    with pytest.raises(ValueError, match="in_modes"):
        tt_conv2d_forward(x, cores, out_modes, in_modes, kernel_size=KERNEL)


def test_forward_rejects_mode_product_mismatch():
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv)

    # kernel を変えると patch 幅が変わり、in_modes の積と一致しない。
    with pytest.raises(ValueError, match="in_modes"):
        tt_conv2d_forward(x, cores, out_modes, in_modes, kernel_size=(3, 3))
    # 出力側 modes が core と合わない（core 側で拒否）。
    with pytest.raises(ValueError):
        tt_conv2d_forward(x, cores, (2, 2, 4), in_modes, kernel_size=KERNEL)


def test_forward_rejects_out_modes_that_disagree_with_cores_product():
    conv, cores, out_modes, in_modes = _cores(2)
    x = _x(conv)
    # core の shape は (2, 4) だが、積が同じ (4, 2) の modes を渡す。
    with pytest.raises(ValueError):
        tt_conv2d_forward(x, cores, (4, 2), in_modes, kernel_size=KERNEL)


def test_forward_rejects_dtype_mismatch():
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv).to(torch.float32)

    with pytest.raises(TypeError, match="dtype"):
        tt_conv2d_forward(x, cores, out_modes, in_modes, kernel_size=KERNEL)
    with pytest.raises(TypeError, match="dtype"):
        tt_conv2d_forward(
            _x(conv), cores, out_modes, in_modes,
            conv.bias.detach().to(torch.float32), kernel_size=KERNEL,
        )


def test_forward_rejects_mixed_core_dtypes():
    conv, cores, out_modes, in_modes = _cores(3)
    mixed = [cores[0], cores[1].to(torch.float32), cores[2]]

    with pytest.raises(TypeError):
        tt_conv2d_forward(_x(conv), mixed, out_modes, in_modes, kernel_size=KERNEL)


def test_forward_rejects_unsupported_dtype_layer_cores():
    conv, cores, out_modes, in_modes = _cores(3)
    halves = [c.to(torch.float16) for c in cores]
    x = _x(conv).to(torch.float16)

    with pytest.raises(TypeError):
        tt_conv2d_forward(x, halves, out_modes, in_modes, kernel_size=KERNEL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDAが利用できません")
def test_forward_rejects_device_mismatch():
    conv, cores, out_modes, in_modes = _cores(3)
    x_cuda = _x(conv).to("cuda")

    with pytest.raises(ValueError, match="device"):
        tt_conv2d_forward(x_cuda, cores, out_modes, in_modes, kernel_size=KERNEL)
    with pytest.raises(ValueError, match="device"):
        tt_conv2d_forward(
            _x(conv), cores, out_modes, in_modes,
            conv.bias.detach().to("cuda"), kernel_size=KERNEL,
        )


def test_forward_rejects_bad_bias_shape():
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(ValueError, match="bias"):
        tt_conv2d_forward(
            _x(conv), cores, out_modes, in_modes, torch.zeros(7, dtype=torch.float64),
            kernel_size=KERNEL,
        )
    with pytest.raises(TypeError, match="bias"):
        tt_conv2d_forward(_x(conv), cores, out_modes, in_modes, [0.0] * 8, kernel_size=KERNEL)


@pytest.mark.parametrize(
    "size, kwargs",
    [
        ((2, 8), {}),  # 高さ 2 < kernel_h 3
        ((9, 1), {}),  # 幅 1 < kernel_w 2
        ((5, 5), {"dilation": 4}),  # 実効 kernel (9, 5) が入力 (5, 5) を超える
    ],
)
def test_forward_rejects_non_positive_output_size(size, kwargs):
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv, size=size)

    with pytest.raises(ValueError, match="出力の空間サイズ"):
        tt_conv2d_forward(x, cores, out_modes, in_modes, kernel_size=KERNEL, **kwargs)


@pytest.mark.parametrize(
    "name, value, error",
    [
        ("stride", 0, ValueError),
        ("stride", -1, ValueError),
        ("stride", (1, 0), ValueError),
        ("stride", (1, 2, 3), ValueError),
        ("stride", (1,), ValueError),
        ("stride", True, TypeError),
        ("stride", (1, True), TypeError),
        ("stride", 1.5, TypeError),
        ("stride", "1", TypeError),
        ("padding", -1, ValueError),
        ("padding", (0, -1), ValueError),
        ("padding", "same", TypeError),
        ("padding", None, TypeError),
        ("dilation", 0, ValueError),
        ("dilation", (1, 0), ValueError),
        ("dilation", 2.0, TypeError),
        ("kernel_size", 0, ValueError),
        ("kernel_size", (3, 0), ValueError),
        ("kernel_size", False, TypeError),
        ("kernel_size", torch.tensor(3), TypeError),
    ],
)
def test_forward_rejects_invalid_conv_settings(name, value, error):
    conv, cores, out_modes, in_modes = _cores(3)
    kwargs = {"kernel_size": KERNEL, name: value}

    with pytest.raises(error):
        tt_conv2d_forward(_x(conv), cores, out_modes, in_modes, **kwargs)


def test_forward_accepts_int_tuple_list_and_numpy_integer_settings():
    np = pytest.importorskip("numpy")
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv)

    reference = tt_conv2d_forward(
        x, cores, out_modes, in_modes, kernel_size=KERNEL, stride=(2, 2), padding=(1, 1)
    )
    equivalent_settings = [
        {"stride": 2, "padding": 1},
        {"stride": [2, 2], "padding": [1, 1]},
        {"stride": (np.int64(2), np.int64(2)), "padding": np.int64(1)},
    ]
    for settings in equivalent_settings:
        out = tt_conv2d_forward(x, cores, out_modes, in_modes, kernel_size=KERNEL, **settings)
        torch.testing.assert_close(out, reference, rtol=0, atol=0)


def test_forward_requires_kernel_size_keyword():
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(TypeError):
        tt_conv2d_forward(_x(conv), cores, out_modes, in_modes)  # type: ignore[call-arg]


def test_forward_rejects_invalid_cores():
    conv, cores, out_modes, in_modes = _cores(3)
    x = _x(conv)

    with pytest.raises(ValueError):
        tt_conv2d_forward(x, [], out_modes, in_modes, kernel_size=KERNEL)
    with pytest.raises(ValueError):
        tt_conv2d_forward(x, cores[:2], out_modes, in_modes, kernel_size=KERNEL)
    with pytest.raises(ValueError):
        tt_conv2d_forward(x, [c[..., 0] for c in cores], out_modes, in_modes, kernel_size=KERNEL)


# ---------------------------------------------------------------------------
# NaN / inf の契約
# ---------------------------------------------------------------------------


def test_forward_propagates_non_finite_input_without_raising():
    """forward は NaN / inf を検査しない。IEEE の規則で出力へ伝わる（nn.Conv2d と同じ）。"""
    conv, layer = _layer(3, padding=1)
    x = _x(conv, size=(6, 6))
    x[0, 0, 2, 2] = float("nan")
    x[1, 1, 3, 3] = float("inf")

    out = layer(x)

    assert out.shape == conv(x).shape
    assert torch.isnan(out[0]).any()
    assert not torch.isfinite(out[1]).all()
    # 影響を受けない領域は有限のまま（画像全体が汚染されるわけではない）。
    assert torch.isfinite(out[0]).any() and torch.isfinite(out[1]).any()


def test_build_rejects_non_finite_weight_with_value_error():
    for bad in (float("nan"), float("inf"), float("-inf")):
        conv = _conv()
        with torch.no_grad():
            conv.weight[0, 0, 0, 0] = bad
        with pytest.raises(ValueError, match="NaN / inf"):
            build_tt_conv2d(conv, *MODES[3])


def test_build_with_non_finite_bias_copies_it_faithfully():
    conv = _conv()
    with torch.no_grad():
        conv.bias[0] = float("nan")

    layer = build_tt_conv2d(conv, *MODES[3])

    assert bool(torch.isnan(layer.bias[0]))


# ---------------------------------------------------------------------------
# TTConv2d の構造
# ---------------------------------------------------------------------------


def test_layer_holds_cores_and_bias_but_no_dense_weight():
    conv, layer = _layer(3)

    assert isinstance(layer, nn.Module)
    assert isinstance(layer.cores, nn.ParameterList)
    assert len(layer.cores) == 3
    assert isinstance(layer.bias, nn.Parameter)
    assert not hasattr(layer, "weight")
    assert set(dict(layer.named_parameters())) == {"cores.0", "cores.1", "cores.2", "bias"}
    assert all(core.requires_grad for core in layer.cores) and layer.bias.requires_grad
    assert all(core.is_leaf for core in layer.cores)


def test_layer_attributes_follow_conv2d_conventions():
    conv = _conv(stride=(2, 1), padding=(1, 2), dilation=(1, 2))
    layer = build_tt_conv2d(conv, *MODES[3])

    assert layer.in_channels == IN_CHANNELS and layer.out_channels == OUT_CHANNELS
    assert layer.kernel_size == KERNEL
    assert layer.stride == (2, 1)
    assert layer.padding == (1, 2)
    assert layer.dilation == (1, 2)
    assert layer.groups == 1
    assert layer.padding_mode == "zeros"
    assert layer.out_modes == MODES[3][0] and layer.in_modes == MODES[3][1]
    for value in (layer.kernel_size, layer.stride, layer.padding, layer.dilation):
        assert isinstance(value, tuple) and all(type(v) is int for v in value)


def test_layer_normalizes_int_settings_to_pairs():
    conv, cores, out_modes, in_modes = _cores(3)
    layer = _manual_layer(cores, out_modes, in_modes, kernel_size=(3, 2), stride=2, padding=1)

    assert layer.stride == (2, 2) and layer.padding == (1, 1) and layer.dilation == (1, 1)


def test_layer_owns_independent_copies_of_cores_and_bias():
    conv, cores, out_modes, in_modes = _cores(3)
    bias = conv.bias.detach().clone()
    source_cores = [c.clone().requires_grad_(True) for c in cores]
    source_bias = bias.clone().requires_grad_(True)

    layer = _manual_layer(source_cores, out_modes, in_modes, bias=source_bias)

    for core, source in zip(layer.cores, source_cores):
        assert core is not source
        assert core.data_ptr() != source.data_ptr()
        assert core.is_leaf and core.grad_fn is None
        torch.testing.assert_close(core.detach(), source.detach(), rtol=0, atol=0)
    assert layer.bias.data_ptr() != source_bias.data_ptr()

    with torch.no_grad():
        layer.cores[0].add_(1.0)
        layer.bias.add_(1.0)
    assert torch.equal(source_cores[0].detach(), cores[0])
    assert torch.equal(source_bias.detach(), bias)

    layer(_x(conv)).sum().backward()
    assert all(c.grad is None for c in source_cores)
    assert source_bias.grad is None


def test_layer_does_not_mutate_arguments():
    conv, cores, out_modes, in_modes = _cores(3)
    saved = [c.clone() for c in cores]

    _manual_layer(cores, out_modes, in_modes)

    for core, saved_core in zip(cores, saved):
        assert torch.equal(core, saved_core)


def test_layer_extra_repr_shows_settings_modes_and_ranks():
    conv, layer = _layer(3, max_rank=3, stride=2, padding=1)
    text = repr(layer)

    assert "in_channels=4" in text and "out_channels=8" in text
    assert "kernel_size=(3, 2)" in text
    assert "stride=(2, 2)" in text and "padding=(1, 1)" in text
    assert "in_modes=(3, 2, 4)" in text and "out_modes=(2, 2, 2)" in text
    assert f"ranks={tt_matrix_ranks(list(layer.cores))}" in text
    assert "bias=True" in text


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_layer_to_dtype_and_device_work_like_nn_module(dtype, device):
    conv, layer = _layer(3, dtype=torch.float64, padding=1)
    other = torch.float32 if dtype == torch.float64 else torch.float64

    moved = layer.to(device=device, dtype=dtype)
    x = _x(conv).to(device=device, dtype=dtype)

    assert moved is layer
    assert all(c.dtype == dtype and c.device.type == torch.device(device).type for c in layer.cores)
    assert layer.bias.dtype == dtype
    assert layer(x).dtype == dtype
    assert isinstance(layer.cores, nn.ParameterList)
    # 戻して別 dtype も通る。
    layer.to(dtype=other)
    assert layer(x.to(other)).dtype == other


def test_layer_forward_rejects_input_dtype_that_differs_from_cores():
    conv, layer = _layer(3, dtype=torch.float64, padding=1)

    with pytest.raises(TypeError, match="dtype"):
        layer(_x(conv).to(torch.float32))


def test_layer_forward_rejects_wrong_channel_count():
    conv, layer = _layer(3)

    with pytest.raises(ValueError):
        layer(torch.randn(2, 3, 9, 8, dtype=torch.float64))


def test_layer_state_dict_roundtrip_reproduces_output():
    conv, layer = _layer(3, max_rank=3, padding=1)
    x = _x(conv)
    clone = _manual_layer(
        [c.detach() for c in layer.cores], layer.out_modes, layer.in_modes,
        bias=layer.bias.detach(), padding=1,
    )
    with torch.no_grad():
        for core in clone.cores:
            core.zero_()
    clone.load_state_dict(layer.state_dict())

    torch.testing.assert_close(clone(x), layer(x), rtol=0, atol=0)


def test_layer_is_trainable_with_optimizer():
    conv, layer = _layer(3, max_rank=3, padding=1)
    x = _x(conv)
    target = torch.randn_like(layer(x))
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in layer.parameters()]

    optimizer.zero_grad()
    F.mse_loss(layer(x), target).backward()
    optimizer.step()

    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, layer.parameters()))


# ---------------------------------------------------------------------------
# TTConv2d コンストラクタの検証
# ---------------------------------------------------------------------------


def test_layer_rejects_groups_other_than_one():
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(NotImplementedError, match="groups"):
        _manual_layer(cores, out_modes, in_modes, groups=2)
    with pytest.raises(ValueError):
        _manual_layer(cores, out_modes, in_modes, groups=0)
    with pytest.raises(TypeError):
        _manual_layer(cores, out_modes, in_modes, groups=True)


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular", "ZEROS"])
def test_layer_rejects_non_zero_padding_mode(mode):
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(NotImplementedError, match="padding_mode"):
        _manual_layer(cores, out_modes, in_modes, padding_mode=mode)
    with pytest.raises(TypeError):
        _manual_layer(cores, out_modes, in_modes, padding_mode=None)


def test_layer_rejects_mode_product_and_channel_mismatches():
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(ValueError, match="out_channels"):
        _manual_layer(cores, out_modes, in_modes, out_channels=OUT_CHANNELS + 1)
    with pytest.raises(ValueError, match="in_channels"):
        _manual_layer(cores, out_modes, in_modes, in_channels=IN_CHANNELS + 1)
    with pytest.raises(ValueError, match="in_channels"):
        _manual_layer(cores, out_modes, in_modes, kernel_size=(3, 3))


@pytest.mark.parametrize("name", ["in_channels", "out_channels"])
@pytest.mark.parametrize("value", [0, -1])
def test_layer_rejects_non_positive_channels(name, value):
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(ValueError):
        _manual_layer(cores, out_modes, in_modes, **{name: value})


@pytest.mark.parametrize("name", ["in_channels", "out_channels"])
@pytest.mark.parametrize("value", [True, 4.0, "4", None])
def test_layer_rejects_non_integer_channels(name, value):
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(TypeError):
        _manual_layer(cores, out_modes, in_modes, **{name: value})


def test_layer_rejects_bad_cores_modes_and_bias():
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(ValueError):
        _manual_layer([], out_modes, in_modes)
    with pytest.raises(ValueError):
        _manual_layer(cores, (2, 2), in_modes)
    with pytest.raises(ValueError):
        _manual_layer(cores, out_modes, (3, 2, 5))
    with pytest.raises(TypeError):
        _manual_layer(cores, out_modes, in_modes, bias=[0.0] * 8)
    with pytest.raises(ValueError):
        _manual_layer(cores, out_modes, in_modes, bias=torch.zeros(7, dtype=torch.float64))
    with pytest.raises(TypeError):
        _manual_layer(cores, out_modes, in_modes, bias=torch.zeros(8, dtype=torch.float32))
    with pytest.raises(TypeError):
        _manual_layer(
            [c.to(torch.float16) for c in cores], out_modes, in_modes
        )


def test_layer_rejects_string_padding():
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(TypeError):
        _manual_layer(cores, out_modes, in_modes, padding="same")


# ---------------------------------------------------------------------------
# build_tt_conv2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
def test_build_preserves_dtype_device_and_bias_presence(dtype, device, bias):
    conv, layer = _layer(3, bias=bias, dtype=dtype, device=device)

    for core in layer.cores:
        assert core.dtype == dtype and core.device == conv.weight.device
    assert (layer.bias is None) == (not bias)
    if bias:
        assert layer.bias.dtype == dtype and layer.bias.device == conv.weight.device
        torch.testing.assert_close(layer.bias.detach(), conv.bias.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_build_does_not_share_storage_with_source_conv(d):
    conv, layer = _layer(d)
    weight_ptr = conv.weight.untyped_storage().data_ptr()
    bias_ptr = conv.bias.untyped_storage().data_ptr()

    for core in layer.cores:
        assert core.untyped_storage().data_ptr() != weight_ptr
    assert layer.bias.untyped_storage().data_ptr() != bias_ptr

    saved_weight = conv.weight.detach().clone()
    with torch.no_grad():
        for core in layer.cores:
            core.zero_()
        layer.bias.zero_()
    assert torch.equal(conv.weight.detach(), saved_weight)
    assert float(conv.bias.detach().abs().sum()) > 0.0


def test_build_does_not_modify_source_conv():
    conv = _conv(padding=1)
    before = copy.deepcopy(conv.state_dict())
    training = conv.training

    build_tt_conv2d(conv, *MODES[3], max_rank=2)

    for name, value in conv.state_dict().items():
        assert torch.equal(value, before[name])
    assert conv.training == training
    assert conv.weight.grad is None
    assert conv.weight.requires_grad and conv.bias.requires_grad


@pytest.mark.parametrize("weight_grad", [True, False])
@pytest.mark.parametrize("bias_grad", [True, False])
def test_build_preserves_requires_grad_for_weight_and_bias_separately(weight_grad, bias_grad):
    conv = _conv()
    conv.weight.requires_grad_(weight_grad)
    conv.bias.requires_grad_(bias_grad)

    layer = build_tt_conv2d(conv, *MODES[3])

    assert all(core.requires_grad is weight_grad for core in layer.cores)
    assert layer.bias.requires_grad is bias_grad


@pytest.mark.parametrize("training", [True, False])
def test_build_preserves_training_flag(training):
    conv = _conv()
    conv.train(training)

    layer = build_tt_conv2d(conv, *MODES[3])

    assert layer.training is training


def test_build_max_rank_none_is_exact_and_integer_caps_ranks():
    conv = _conv()
    exact = build_tt_conv2d(conv, *MODES[3], max_rank=None)
    limited = build_tt_conv2d(conv, *MODES[3], max_rank=2)

    assert max(tt_matrix_ranks(list(limited.cores))) <= 2
    assert max(tt_matrix_ranks(list(exact.cores))) > 2
    default = build_tt_conv2d(conv, *MODES[3])
    assert tt_matrix_ranks(list(default.cores)) == tt_matrix_ranks(list(exact.cores))


def test_build_d1_ignores_max_rank_but_validates_it():
    conv = _conv()
    layer = build_tt_conv2d(conv, *MODES[1], max_rank=1)

    assert len(layer.cores) == 1
    torch.testing.assert_close(layer(_x(conv)), conv(_x(conv)), **_tol(torch.float64))
    with pytest.raises(ValueError):
        build_tt_conv2d(conv, *MODES[1], max_rank=0)


@pytest.mark.parametrize("max_rank", [0, -1])
def test_build_rejects_non_positive_max_rank(max_rank):
    with pytest.raises(ValueError):
        build_tt_conv2d(_conv(), *MODES[3], max_rank=max_rank)


@pytest.mark.parametrize("max_rank", [True, 2.0, "2"])
def test_build_rejects_non_integer_max_rank(max_rank):
    with pytest.raises(TypeError):
        build_tt_conv2d(_conv(), *MODES[3], max_rank=max_rank)


def test_build_uses_dense_to_tt_matrix_cores_on_flattened_weight(monkeypatch):
    conv = _conv()
    calls = []
    real = tt_conv_module.dense_to_tt_matrix_cores

    def spy(W, out_modes, in_modes, *, max_rank=None):
        calls.append((tuple(W.shape), max_rank))
        return real(W, out_modes, in_modes, max_rank=max_rank)

    monkeypatch.setattr(tt_conv_module, "dense_to_tt_matrix_cores", spy)
    build_tt_conv2d(conv, *MODES[3], max_rank=3)

    assert calls == [((OUT_CHANNELS, IN_CHANNELS * 3 * 2), 3)]


def test_build_rejects_non_conv_modules():
    with pytest.raises(TypeError):
        build_tt_conv2d(nn.Linear(24, 8), *MODES[3])
    with pytest.raises(TypeError):
        build_tt_conv2d(torch.zeros(8, 4, 3, 2), *MODES[3])
    with pytest.raises(TypeError):
        build_tt_conv2d(nn.ConvTranspose2d(4, 8, 3), (2, 2, 2), (3, 3, 4))


def test_build_rejects_groups_other_than_one():
    conv = nn.Conv2d(4, 8, KERNEL, groups=2, dtype=torch.float64)

    with pytest.raises(NotImplementedError, match="groups"):
        build_tt_conv2d(conv, (2, 2, 2), (3, 2, 2))


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_build_rejects_non_zero_padding_modes(mode):
    conv = nn.Conv2d(4, 8, KERNEL, padding=1, padding_mode=mode, dtype=torch.float64)

    with pytest.raises(NotImplementedError, match="padding_mode"):
        build_tt_conv2d(conv, *MODES[3])


@pytest.mark.parametrize("padding", ["same", "valid"])
def test_build_rejects_string_padding(padding):
    conv = nn.Conv2d(4, 8, 3, padding=padding, dtype=torch.float64)

    with pytest.raises(NotImplementedError, match="padding"):
        build_tt_conv2d(conv, (2, 2, 2), (3, 3, 4))


def test_build_rejects_mode_and_channel_mismatch():
    conv = _conv()

    with pytest.raises(ValueError, match="out_channels"):
        build_tt_conv2d(conv, (2, 2, 3), (3, 2, 4))
    with pytest.raises(ValueError, match="in_channels"):
        build_tt_conv2d(conv, (2, 2, 2), (3, 2, 5))
    with pytest.raises(ValueError):
        build_tt_conv2d(conv, (2, 2, 2), (3, 2))  # site 数が違う
    with pytest.raises(TypeError):
        build_tt_conv2d(conv, (2, True, 4), (3, 2, 4))


def test_build_rejects_unsupported_weight_dtype():
    conv = nn.Conv2d(4, 8, KERNEL, dtype=torch.float16)

    with pytest.raises(TypeError):
        build_tt_conv2d(conv, *MODES[3])


@pytest.mark.parametrize("dtype", DTYPES)
def test_build_then_forward_end_to_end_float32_float64(dtype):
    conv, layer = _layer(3, dtype=dtype, padding=1)
    x = _x(conv)

    torch.testing.assert_close(layer(x), conv(x), **_tol(dtype))


# ---------------------------------------------------------------------------
# factorize_named_tt_conv2d
# ---------------------------------------------------------------------------


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 3, padding=1, dtype=torch.float64)
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 8, KERNEL, padding=1, dtype=torch.float64),
        )
        self.encoder = nn.ModuleDict(
            {"inner": nn.Sequential(nn.Conv2d(8, 4, 1, dtype=torch.float64))}
        )
        self.fc = nn.Linear(4, 2, dtype=torch.float64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block(x)
        x = self.encoder["inner"](x)
        return self.fc(x.mean(dim=(2, 3)))


def test_factorize_named_replaces_nested_sequential_layer_only():
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 2, 7, 6, dtype=torch.float64)

    compressed = factorize_named_tt_conv2d(model, "block.1", (2, 2, 2), (2, 3, 4))

    assert compressed is not model
    assert isinstance(compressed.block[1], TTConv2d)
    assert isinstance(compressed.block[0], nn.ReLU)
    assert isinstance(compressed.conv1, nn.Conv2d)
    assert isinstance(model.block[1], nn.Conv2d)
    torch.testing.assert_close(compressed(x), model(x), rtol=1e-9, atol=1e-10)


def test_factorize_named_supports_direct_and_module_dict_names():
    torch.manual_seed(1)
    model = _Net()

    direct = factorize_named_tt_conv2d(model, "conv1", (2, 2), (2, 9))
    nested = factorize_named_tt_conv2d(model, "encoder.inner.0", (2, 2), (2, 4))

    assert isinstance(direct.conv1, TTConv2d)
    assert isinstance(nested.encoder["inner"][0], TTConv2d)
    assert direct.conv1.kernel_size == (3, 3) and direct.conv1.padding == (1, 1)
    assert nested.encoder["inner"][0].kernel_size == (1, 1)


def test_factorize_named_does_not_touch_original_model_or_share_parameters():
    torch.manual_seed(2)
    model = _Net()
    before = copy.deepcopy(model.state_dict())
    training_flags = [m.training for m in model.modules()]

    compressed = factorize_named_tt_conv2d(model, "block.1", (2, 2, 2), (2, 3, 4), max_rank=2)

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert [m.training for m in model.modules()] == training_flags
    original_pointers = {p.data_ptr() for p in model.parameters()}
    assert all(p.data_ptr() not in original_pointers for p in compressed.parameters())

    with torch.no_grad():
        for p in compressed.parameters():
            p.zero_()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_factorize_named_keeps_model_training_state_per_module():
    model = _Net()
    model.eval()
    model.block[1].train()

    compressed = factorize_named_tt_conv2d(model, "block.1", (2, 2, 2), (2, 3, 4))

    assert compressed.training is False
    assert compressed.conv1.training is False
    assert compressed.block[1].training is True  # 置換後の層は元の Conv2d に揃う


def test_factorize_named_max_rank_passed_through():
    model = _Net()

    compressed = factorize_named_tt_conv2d(model, "block.1", (2, 2, 2), (2, 3, 4), max_rank=2)

    assert max(tt_matrix_ranks(list(compressed.block[1].cores))) <= 2


@pytest.mark.parametrize("name", ["fc", "block.0", "block", ""])
def test_factorize_named_rejects_non_conv_targets(name):
    model = _Net()

    if name == "":
        with pytest.raises(ValueError):
            factorize_named_tt_conv2d(model, name, (2, 2, 2), (2, 3, 4))
    else:
        with pytest.raises(TypeError, match="Conv2d"):
            factorize_named_tt_conv2d(model, name, (2, 2, 2), (2, 3, 4))


def test_factorize_named_rejects_bad_layer_names():
    model = _Net()

    with pytest.raises(TypeError):
        factorize_named_tt_conv2d(model, 1, (2, 2, 2), (2, 3, 4))  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        factorize_named_tt_conv2d(model, "missing", (2, 2, 2), (2, 3, 4))


def test_factorize_named_propagates_unsupported_conv_settings():
    model = nn.Sequential(nn.Conv2d(4, 8, 3, groups=2, dtype=torch.float64))

    with pytest.raises(NotImplementedError):
        factorize_named_tt_conv2d(model, "0", (2, 2, 2), (3, 3, 2))


def test_factorize_named_reuses_build_tt_conv2d(monkeypatch):
    model = _Net()
    calls = []
    real = tt_conv_module.build_tt_conv2d

    def spy(conv, out_modes, in_modes, *, max_rank=None):
        calls.append(max_rank)
        return real(conv, out_modes, in_modes, max_rank=max_rank)

    monkeypatch.setattr(tt_conv_module, "build_tt_conv2d", spy)
    factorize_named_tt_conv2d(model, "block.1", (2, 2, 2), (2, 3, 4), max_rank=5)

    assert calls == [5]


# ---------------------------------------------------------------------------
# tt_conv2d_to_dense_weight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_to_dense_weight_roundtrips_exact_tt(d, dtype):
    conv, layer = _layer(d, dtype=dtype)

    weight = tt_conv2d_to_dense_weight(
        list(layer.cores), layer.out_modes, layer.in_modes,
        in_channels=IN_CHANNELS, kernel_size=KERNEL,
    )

    assert weight.shape == conv.weight.shape == (OUT_CHANNELS, IN_CHANNELS, 3, 2)
    assert weight.dtype == dtype
    torch.testing.assert_close(weight, conv.weight.detach(), **_tol(dtype))


def test_to_dense_weight_reuses_tt_matrix_to_dense(monkeypatch):
    conv, cores, out_modes, in_modes = _cores(3)
    calls = []
    real = tt_conv_module.tt_matrix_to_dense

    def spy(cores_arg, out_arg, in_arg):
        calls.append(len(cores_arg))
        return real(cores_arg, out_arg, in_arg)

    monkeypatch.setattr(tt_conv_module, "tt_matrix_to_dense", spy)
    tt_conv2d_to_dense_weight(cores, out_modes, in_modes, in_channels=4, kernel_size=KERNEL)

    assert calls == [3]


def test_to_dense_weight_keeps_autograd_graph_and_does_not_mutate_cores():
    conv, cores, out_modes, in_modes = _cores(3, max_rank=2)
    leaves = [c.clone().requires_grad_(True) for c in cores]
    saved = [c.detach().clone() for c in leaves]

    weight = tt_conv2d_to_dense_weight(
        leaves, out_modes, in_modes, in_channels=4, kernel_size=KERNEL
    )
    assert weight.requires_grad
    weight.square().sum().backward()

    for leaf, saved_core in zip(leaves, saved):
        assert torch.equal(leaf.detach(), saved_core)
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_to_dense_weight_d1_does_not_share_storage_with_core():
    conv, cores, out_modes, in_modes = _cores(1)

    weight = tt_conv2d_to_dense_weight(
        cores, out_modes, in_modes, in_channels=4, kernel_size=KERNEL
    )
    saved = cores[0].clone()
    weight.zero_()

    assert torch.equal(cores[0], saved)


@pytest.mark.parametrize("device", DEVICES)
def test_to_dense_weight_keeps_device(device):
    conv, cores, out_modes, in_modes = _cores(3, device=device)

    weight = tt_conv2d_to_dense_weight(
        cores, out_modes, in_modes, in_channels=4, kernel_size=KERNEL
    )

    assert weight.device == cores[0].device


def test_to_dense_weight_rejects_inconsistent_arguments():
    conv, cores, out_modes, in_modes = _cores(3)

    with pytest.raises(ValueError, match="in_channels"):
        tt_conv2d_to_dense_weight(cores, out_modes, in_modes, in_channels=5, kernel_size=KERNEL)
    with pytest.raises(ValueError, match="in_channels"):
        tt_conv2d_to_dense_weight(cores, out_modes, in_modes, in_channels=4, kernel_size=(3, 3))
    with pytest.raises(ValueError):
        tt_conv2d_to_dense_weight(cores, (2, 2, 4), in_modes, in_channels=4, kernel_size=KERNEL)
    with pytest.raises(ValueError):
        tt_conv2d_to_dense_weight([], out_modes, in_modes, in_channels=4, kernel_size=KERNEL)
    with pytest.raises(TypeError):
        tt_conv2d_to_dense_weight(cores, out_modes, in_modes, in_channels=True, kernel_size=KERNEL)
    with pytest.raises(ValueError):
        tt_conv2d_to_dense_weight(cores, out_modes, in_modes, in_channels=0, kernel_size=KERNEL)


# ---------------------------------------------------------------------------
# sweep_tt_conv2d_weight_ranks
# ---------------------------------------------------------------------------


RANKS = [4, 1, 2, 2, 100]


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sweep_matches_matrix_sweep_and_adds_layer_level_fields(dtype, bias):
    conv = _conv(bias=bias, dtype=dtype)
    out_modes, in_modes = MODES[3]

    records = sweep_tt_conv2d_weight_ranks(conv, out_modes, in_modes, RANKS)
    matrix_records = sweep_tt_matrix_weight_ranks(
        conv.weight.detach().reshape(OUT_CHANNELS, -1), out_modes, in_modes, RANKS
    )

    bias_parameters = OUT_CHANNELS if bias else 0
    assert len(records) == len(RANKS)
    for record, matrix_record, rank in zip(records, matrix_records, RANKS):
        assert record["requested_rank"] == rank == matrix_record["requested_rank"]
        assert record["actual_ranks"] == matrix_record["actual_ranks"]
        assert record["tt_core_parameters"] == matrix_record["tt_parameters"]
        assert record["dense_weight_parameters"] == matrix_record["dense_parameters"]
        assert record["dense_weight_parameters"] == conv.weight.numel()
        assert record["bias_parameters"] == bias_parameters
        assert record["dense_layer_parameters"] == conv.weight.numel() + bias_parameters
        assert record["tt_layer_parameters"] == matrix_record["tt_parameters"] + bias_parameters
        assert record["weight_compression_ratio"] == matrix_record["compression_ratio"]
        assert record["layer_compression_ratio"] == pytest.approx(
            record["dense_layer_parameters"] / record["tt_layer_parameters"]
        )
        assert record["absolute_error"] == matrix_record["absolute_error"]
        assert record["relative_error"] == matrix_record["relative_error"]
        assert record["is_weight_compressed"] == matrix_record["is_compressed"]
        assert record["is_layer_compressed"] == (
            record["tt_layer_parameters"] < record["dense_layer_parameters"]
        )


def test_sweep_record_schema_and_python_types():
    conv = _conv()

    records = sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [2])

    assert set(records[0]) == {
        "requested_rank", "actual_ranks", "tt_core_parameters", "dense_weight_parameters",
        "bias_parameters", "dense_layer_parameters", "tt_layer_parameters",
        "weight_compression_ratio", "layer_compression_ratio", "absolute_error",
        "relative_error", "is_weight_compressed", "is_layer_compressed",
    }
    record = records[0]
    assert type(record["requested_rank"]) is int
    assert type(record["actual_ranks"]) is tuple
    for key in (
        "tt_core_parameters", "dense_weight_parameters", "bias_parameters",
        "dense_layer_parameters", "tt_layer_parameters",
    ):
        assert type(record[key]) is int
    for key in ("weight_compression_ratio", "layer_compression_ratio", "absolute_error", "relative_error"):
        assert type(record[key]) is float
    assert type(record["is_weight_compressed"]) is bool
    assert type(record["is_layer_compressed"]) is bool


def test_sweep_preserves_order_and_duplicate_ranks():
    conv = _conv()

    records = sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [3, 1, 3, 2, 1])

    assert [r["requested_rank"] for r in records] == [3, 1, 3, 2, 1]
    assert records[0] == records[2] and records[1] == records[4]


def test_sweep_full_rank_is_not_a_compression_when_cores_exceed_dense():
    conv = _conv()

    records = sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [1, 100])

    assert records[0]["is_weight_compressed"] is True
    assert records[1]["is_weight_compressed"] is False
    assert records[1]["relative_error"] < 1e-10


def test_sweep_does_not_modify_conv_or_build_graph():
    conv = _conv()
    before = copy.deepcopy(conv.state_dict())

    records = sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [2, 3])

    for name, value in conv.state_dict().items():
        assert torch.equal(value, before[name])
    assert conv.weight.grad is None and conv.bias.grad is None
    assert all(not isinstance(v, torch.Tensor) for r in records for v in r.values())


def test_sweep_matches_build_parameter_counts():
    conv = _conv()

    for record in sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [1, 2, 3]):
        layer = build_tt_conv2d(conv, *MODES[3], max_rank=record["requested_rank"])
        assert record["actual_ranks"] == tt_matrix_ranks(list(layer.cores))
        assert record["tt_core_parameters"] == tt_matrix_num_parameters(list(layer.cores))
        assert record["tt_layer_parameters"] == sum(p.numel() for p in layer.parameters())


def test_sweep_reuses_matrix_sweep(monkeypatch):
    conv = _conv()
    calls = []
    real = tt_conv_module.sweep_tt_matrix_weight_ranks

    def spy(W, out_modes, in_modes, ranks):
        calls.append((tuple(W.shape), list(ranks)))
        return real(W, out_modes, in_modes, ranks)

    monkeypatch.setattr(tt_conv_module, "sweep_tt_matrix_weight_ranks", spy)
    sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [2, 1])

    assert calls == [((OUT_CHANNELS, 24), [2, 1])]


def test_sweep_rejects_invalid_inputs():
    conv = _conv()

    with pytest.raises(TypeError):
        sweep_tt_conv2d_weight_ranks(nn.Linear(24, 8), *MODES[3], [1])
    with pytest.raises(NotImplementedError):
        sweep_tt_conv2d_weight_ranks(
            nn.Conv2d(4, 8, KERNEL, groups=2, dtype=torch.float64), (2, 2, 2), (3, 2, 2), [1]
        )
    with pytest.raises(ValueError):
        sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [])
    with pytest.raises(ValueError):
        sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [0])
    with pytest.raises(TypeError):
        sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [True])
    with pytest.raises(TypeError):
        sweep_tt_conv2d_weight_ranks(conv, *MODES[3], "12")
    with pytest.raises(ValueError, match="out_channels"):
        sweep_tt_conv2d_weight_ranks(conv, (2, 2, 3), (3, 2, 4), [1])


def test_sweep_rejects_zero_and_non_finite_weight():
    zero = _conv()
    with torch.no_grad():
        zero.weight.zero_()
    with pytest.raises(ValueError, match="ゼロ"):
        sweep_tt_conv2d_weight_ranks(zero, *MODES[3], [1])

    for bad in (float("nan"), float("inf")):
        conv = _conv()
        with torch.no_grad():
            conv.weight[0, 0, 0, 0] = bad
        with pytest.raises(ValueError, match="NaN / inf"):
            sweep_tt_conv2d_weight_ranks(conv, *MODES[3], [1])


@pytest.mark.parametrize("d", [1, 2])
def test_sweep_supports_other_site_counts(d):
    conv = _conv()

    records = sweep_tt_conv2d_weight_ranks(conv, *MODES[d], [1, 2])

    assert len(records) == 2
    assert all(len(r["actual_ranks"]) == d + 1 for r in records)


def test_sweep_total_parameter_arithmetic_for_a_pretrained_like_conv2():
    """Notebook 03 の conv2（64x32x3x3, modes (4,4,4)/(6,6,8)）と同じ数値。"""
    conv = nn.Conv2d(32, 64, 3, dtype=torch.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        conv.weight.normal_()

    records = sweep_tt_conv2d_weight_ranks(conv, (4, 4, 4), (6, 6, 8), [32])

    assert records[0]["actual_ranks"] == (1, 24, 32, 1)
    assert records[0]["tt_core_parameters"] == 20_032
    assert records[0]["dense_weight_parameters"] == 18_432
    assert records[0]["bias_parameters"] == 64
    assert math.isclose(
        records[0]["weight_compression_ratio"], 18_432 / 20_032, rel_tol=1e-12
    )
