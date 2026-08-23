"""Tucker-2 Conv 共通化の回帰テスト。

参照元 Notebook:
    notebooks/20_tucker/10_cifar10_cnn/01_tucker2_conv.ipynb
"""

import pytest
import torch
from torch import nn

from nn_compression.compression import (
    build_tucker2_conv,
    tucker2_decompose_conv_weight,
    tucker_parameter_count,
)
from nn_compression.metrics import relative_frobenius_error


def _conv2d(
    *,
    in_channels: int = 32,
    out_channels: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    bias: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Conv2d:
    device = device or torch.device("cpu")
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    ).to(device=device, dtype=dtype)
    torch.manual_seed(0)
    with torch.no_grad():
        conv.weight.normal_()
        if bias:
            conv.bias.normal_()
    return conv


def test_tucker2_decompose_conv_weight_shapes():
    conv = _conv2d()
    rank_out, rank_in = 32, 16
    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), rank_out, rank_in
    )
    k_h, k_w = conv.kernel_size
    assert tuple(core.shape) == (rank_out, rank_in, k_h, k_w)
    assert tuple(u_out.shape) == (conv.out_channels, rank_out)
    assert tuple(u_in.shape) == (conv.in_channels, rank_in)


def test_build_tucker2_conv_returns_three_layer_sequential():
    conv = _conv2d()
    tucker_conv = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    assert isinstance(tucker_conv, nn.Sequential)
    assert len(tucker_conv) == 3
    assert all(isinstance(layer, nn.Conv2d) for layer in tucker_conv)


def test_build_tucker2_conv_weight_shapes():
    conv = _conv2d(in_channels=32, out_channels=64, kernel_size=3)
    rank_out, rank_in = 32, 16
    input_layer, core_layer, output_layer = build_tucker2_conv(
        conv, rank_out, rank_in
    )

    assert tuple(input_layer.weight.shape) == (rank_in, conv.in_channels, 1, 1)
    assert tuple(core_layer.weight.shape) == (
        rank_out,
        rank_in,
        conv.kernel_size[0],
        conv.kernel_size[1],
    )
    assert tuple(output_layer.weight.shape) == (conv.out_channels, rank_out, 1, 1)


def test_build_tucker2_conv_output_shape_matches_original():
    conv = _conv2d()
    tucker_conv = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    x = torch.randn(4, conv.in_channels, 16, 16)
    with torch.no_grad():
        y_original = conv(x)
        y_tucker = tucker_conv(x)
    assert y_original.shape == y_tucker.shape


def test_build_tucker2_conv_core_layer_inherits_spatial_config():
    conv = _conv2d(stride=2, padding=2, dilation=2)
    _, core_layer, _ = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    assert core_layer.stride == conv.stride
    assert core_layer.padding == conv.padding
    assert core_layer.dilation == conv.dilation
    assert core_layer.padding_mode == conv.padding_mode


def test_build_tucker2_conv_side_layers_use_1x1_without_spatial_stride():
    conv = _conv2d(stride=2, padding=2)
    input_layer, _, output_layer = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    assert input_layer.kernel_size == (1, 1)
    assert output_layer.kernel_size == (1, 1)
    assert input_layer.stride == (1, 1)
    assert output_layer.stride == (1, 1)
    assert input_layer.padding == (0, 0)
    assert output_layer.padding == (0, 0)


def test_build_tucker2_conv_without_bias():
    conv = _conv2d(bias=False)
    _, _, output_layer = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    assert output_layer.bias is None


def test_build_tucker2_conv_with_bias_copied_to_output_layer():
    conv = _conv2d(bias=True)
    _, _, output_layer = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    assert output_layer.bias is not None
    assert torch.equal(output_layer.bias, conv.bias)


def test_build_tucker2_conv_preserves_device_and_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA が利用できません")
    conv = _conv2d(device=torch.device("cuda"), dtype=torch.float64)
    tucker_conv = build_tucker2_conv(conv, rank_out=32, rank_in=16)
    for layer in tucker_conv:
        assert layer.weight.device == conv.weight.device
        assert layer.weight.dtype == conv.weight.dtype
        if layer.bias is not None:
            assert layer.bias.device == conv.weight.device
            assert layer.bias.dtype == conv.weight.dtype


def test_build_tucker2_conv_rejects_grouped_conv():
    conv = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=2)
    with pytest.raises(ValueError, match="groups=1"):
        build_tucker2_conv(conv, rank_out=32, rank_in=16)


def test_build_tucker2_conv_does_not_mutate_original_conv():
    conv = _conv2d()
    original_weight = conv.weight.detach().clone()
    original_bias = conv.bias.detach().clone()
    build_tucker2_conv(conv, rank_out=32, rank_in=16)
    assert torch.equal(conv.weight, original_weight)
    assert torch.equal(conv.bias, original_bias)


def test_build_tucker2_conv_near_full_rank_has_valid_shapes():
    conv = _conv2d(in_channels=8, out_channels=12, kernel_size=3)
    rank_out, rank_in = conv.out_channels, conv.in_channels
    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), rank_out, rank_in
    )
    k_h, k_w = conv.kernel_size
    assert tuple(core.shape) == (rank_out, rank_in, k_h, k_w)
    assert tuple(u_out.shape) == (conv.out_channels, rank_out)
    assert tuple(u_in.shape) == (conv.in_channels, rank_in)

    tucker_conv = build_tucker2_conv(conv, rank_out, rank_in)
    x = torch.randn(2, conv.in_channels, 8, 8)
    with torch.no_grad():
        y_original = conv(x)
        y_tucker = tucker_conv(x)
    assert y_original.shape == y_tucker.shape
    assert relative_frobenius_error(y_original, y_tucker).item() < 1e-4


def test_build_tucker2_conv_parameter_count_matches_formula():
    conv = _conv2d(in_channels=32, out_channels=64, kernel_size=3)
    rank_out, rank_in = 32, 16
    ranks = {0: rank_out, 1: rank_in}
    tucker_conv = build_tucker2_conv(conv, rank_out, rank_in)

    formula = (
        conv.in_channels * rank_in
        + rank_out * rank_in * conv.kernel_size[0] * conv.kernel_size[1]
        + conv.out_channels * rank_out
    )
    src_count = tucker_parameter_count(conv.weight.shape, ranks)
    module_count = sum(
        layer.weight.numel()
        for layer in tucker_conv
        if isinstance(layer, nn.Conv2d)
    )

    assert formula == src_count
    assert src_count == module_count


def test_tucker2_decompose_rejects_invalid_rank():
    conv = _conv2d()
    with pytest.raises(ValueError):
        tucker2_decompose_conv_weight(conv.weight, rank_out=0, rank_in=16)
    with pytest.raises(ValueError):
        tucker2_decompose_conv_weight(
            conv.weight, rank_out=conv.out_channels + 1, rank_in=16
        )
