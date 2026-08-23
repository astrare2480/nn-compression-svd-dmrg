"""nn.Conv2d に対する Tucker-2 分解。

参照元:
    notebooks/20_tucker/10_cifar10_cnn/01_tucker2_conv.ipynb

Tensor そのものの HOSVD は ``tucker``、Conv2d への適用はここに置く。
"""

from __future__ import annotations

import torch
from torch import nn

from .svd import coerce_rank
from .tucker import hosvd


def _factory_kwargs(conv: nn.Conv2d) -> dict:
    return {
        "device": conv.weight.device,
        "dtype": conv.weight.dtype,
    }


def _unsupported_conv2d(conv: nn.Conv2d) -> None:
    """Tucker-2 Conv は groups=1 の通常 Conv2d のみ対応する。"""
    if conv.groups != 1:
        raise ValueError(
            "groups=1 の Conv2d のみ対応しています: "
            f"groups={conv.groups}"
        )
    if getattr(conv, "transposed", False):
        raise ValueError("転置畳み込みは未対応です。")


def _coerce_tucker2_ranks(
    shape: tuple[int, ...],
    rank_out: int,
    rank_in: int,
) -> tuple[int, int]:
    """Conv2d weight ``(Cout, Cin, kH, kW)`` の Tucker-2 rank を検証する。"""
    cout, cin = shape[0], shape[1]
    return (
        coerce_rank(rank_out, cout, name="rank_out"),
        coerce_rank(rank_in, cin, name="rank_in"),
    )


def tucker2_decompose_conv_weight(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conv2d weight を mode 0 / mode 1 だけ HOSVD し、core と factor を返す。

    weight:
        ``(Cout, Cin, kH, kW)``

    返り値:
        core, U_out, U_in
    """
    rank_out, rank_in = _coerce_tucker2_ranks(
        tuple(weight.shape), rank_out, rank_in
    )
    ranks = {0: rank_out, 1: rank_in}
    core, factors = hosvd(weight, ranks)
    return core, factors[0], factors[1]


def build_tucker2_conv(
    conv: nn.Conv2d,
    rank_out: int,
    rank_in: int,
) -> nn.Sequential:
    """1 つの Conv2d を Tucker-2 分解し、1x1 -> kHxkW -> 1x1 の Sequential を返す。

    conv:
        groups=1 の nn.Conv2d

    rank_out:
        out channel 側の Tucker rank

    rank_in:
        in channel 側の Tucker rank
    """
    _unsupported_conv2d(conv)
    factory = _factory_kwargs(conv)
    has_bias = conv.bias is not None
    shape = conv.weight.shape

    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), rank_out, rank_in
    )

    input_layer = nn.Conv2d(
        in_channels=shape[1],
        out_channels=rank_in,
        kernel_size=1,
        bias=False,
        **factory,
    )

    output_layer = nn.Conv2d(
        in_channels=rank_out,
        out_channels=shape[0],
        kernel_size=1,
        bias=has_bias,
        **factory,
    )

    core_layer = nn.Conv2d(
        in_channels=rank_in,
        out_channels=rank_out,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=False,
        padding_mode=conv.padding_mode,
        **factory,
    )

    with torch.no_grad():
        input_layer.weight.copy_(u_in.T[:, :, None, None])
        core_layer.weight.copy_(core)
        output_layer.weight.copy_(u_out[:, :, None, None])
        if has_bias:
            output_layer.bias.copy_(conv.bias)

    input_layer.weight.requires_grad = conv.weight.requires_grad
    core_layer.weight.requires_grad = conv.weight.requires_grad
    output_layer.weight.requires_grad = conv.weight.requires_grad
    if has_bias:
        output_layer.bias.requires_grad = conv.bias.requires_grad

    return nn.Sequential(input_layer, core_layer, output_layer)
