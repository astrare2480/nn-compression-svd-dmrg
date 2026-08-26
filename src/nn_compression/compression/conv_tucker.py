"""nn.Conv2d に対する Tucker-2 分解。

参照元:
    notebooks/20_tucker/10_cifar10_cnn/01_tucker2_conv.ipynb
    notebooks/20_tucker/10_cifar10_cnn/04_hooi_tucker2.ipynb
    notebooks/20_tucker/10_cifar10_cnn/05_hosvd_vs_hooi_finetuning.ipynb

Tensor そのものの HOSVD / HOOI は ``tucker`` / ``hooi``、Conv2d への適用はここに置く。
"""

from __future__ import annotations

import torch
from torch import nn

from .hooi import hooi, hooi_sweep
from .svd import coerce_rank
from .tucker import hosvd, reconstruct_tucker


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
    """Conv2d weight ``(C_out, C_in, kH, kW)`` の Tucker-2 rank を検証する。"""
    if len(shape) != 4:
        raise ValueError(
            f"Conv2d weight は4階テンソルである必要があります: shape={shape}"
        )
    cout, cin = shape[0], shape[1]
    return (
        coerce_rank(rank_out, cout, name="rank_out"),
        coerce_rank(rank_in, cin, name="rank_in"),
    )


def _validate_tucker2_components(
    conv: nn.Conv2d,
    core: torch.Tensor,
    u_out: torch.Tensor,
    u_in: torch.Tensor,
) -> tuple[int, int]:
    """core / U_out / U_in の shape が Conv2d weight と整合するか検証する。"""
    rank_out, rank_in = _coerce_tucker2_ranks(
        tuple(conv.weight.shape), u_out.shape[1], u_in.shape[1]
    )
    cout, cin = conv.out_channels, conv.in_channels
    k_h, k_w = conv.kernel_size

    if u_out.shape != (cout, rank_out):
        raise ValueError(
            f"u_out.shape={tuple(u_out.shape)} は ({cout}, {rank_out}) "
            "である必要があります。"
        )
    if u_in.shape != (cin, rank_in):
        raise ValueError(
            f"u_in.shape={tuple(u_in.shape)} は ({cin}, {rank_in}) "
            "である必要があります。"
        )
    if core.shape != (rank_out, rank_in, k_h, k_w):
        raise ValueError(
            f"core.shape={tuple(core.shape)} は "
            f"({rank_out}, {rank_in}, {k_h}, {k_w}) である必要があります。"
        )
    return rank_out, rank_in


def build_tucker2_conv_from_components(
    conv: nn.Conv2d,
    core: torch.Tensor,
    u_out: torch.Tensor,
    u_in: torch.Tensor,
) -> nn.Sequential:
    """事前計算した Tucker-2 core / factor から 1x1 -> kHxkW -> 1x1 を構築する。

    Conv weight ``(C_out, C_in, kH, kW)`` に対し、

    - 左 1x1: ``C_in -> rank_in`` （``U_in^T`` を weight に配置）
    - 中 kHxkW: ``rank_in -> rank_out`` （core をそのまま配置）
    - 右 1x1: ``rank_out -> C_out`` （``U_out`` を weight に配置）

    mode 0 が out channel (C_out)、mode 1 が in channel (C_in) に対応する。
    """
    _unsupported_conv2d(conv)
    rank_out, rank_in = _validate_tucker2_components(conv, core, u_out, u_in)
    factory = _factory_kwargs(conv)
    has_bias = conv.bias is not None
    shape = conv.weight.shape

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
    """1 つの Conv2d を HOSVD Tucker-2 分解し、1x1 -> kHxkW -> 1x1 の Sequential を返す。"""
    _unsupported_conv2d(conv)
    core, u_out, u_in = tucker2_decompose_conv_weight(
        conv.weight.detach(), rank_out, rank_in
    )
    return build_tucker2_conv_from_components(conv, core, u_out, u_in)


def tucker2_hooi_sweep(
    weight: torch.Tensor,
    factors: dict[int, torch.Tensor],
    rank_out: int,
    rank_in: int,
) -> dict[int, torch.Tensor]:
    """Conv2d weight の partial Tucker-2 HOOI を 1 sweep 実行する。

    ``ranks={0: rank_out, 1: rank_in}`` として汎用 ``hooi_sweep`` を呼ぶ。
    mode 2 / 3（kH, kW）は ranks に含めないため圧縮されない。
    mode 1 更新時は、この sweep 内で更新済みの ``U_out`` を使う。
    """
    rank_out, rank_in = _coerce_tucker2_ranks(
        tuple(weight.shape), rank_out, rank_in
    )
    return hooi_sweep(weight, factors, {0: rank_out, 1: rank_in})


def tucker2_hooi(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
    max_iter: int = 30,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]:
    """Conv2d weight を HOOI Tucker-2 分解する。

    HOSVD を初期値に ``ranks={0: rank_out, 1: rank_in}`` で反復する。
    返り値 ``history[0]`` は HOSVD 初期誤差。
    """
    rank_out, rank_in = _coerce_tucker2_ranks(
        tuple(weight.shape), rank_out, rank_in
    )
    return hooi(
        weight,
        {0: rank_out, 1: rank_in},
        max_iter=max_iter,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )


def tucker2_effective_weight(seq: nn.Sequential) -> torch.Tensor:
    """Tucker-2 Sequential (1x1 -> kxk -> 1x1) を等価な 4 階 weight に再構成する。

    fine-tuning 前後の weight 誤差比較など、分解表現を 1 つの Conv weight
    に戻して評価するときに使う。
    """
    if len(seq) != 3:
        raise ValueError(
            f"Tucker-2 Sequential は3層である必要があります: len={len(seq)}"
        )
    for layer in seq:
        if not isinstance(layer, nn.Conv2d):
            raise TypeError("Tucker-2 Sequential の各層は nn.Conv2d である必要があります。")

    u_in = seq[0].weight.detach()[:, :, 0, 0].T
    core = seq[1].weight.detach()
    u_out = seq[2].weight.detach()[:, :, 0, 0]
    return reconstruct_tucker(core, {0: u_out, 1: u_in})
