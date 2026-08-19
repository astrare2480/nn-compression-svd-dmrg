"""nn.Conv2d に対する汎用 SVD 処理。

参照元:
    notebooks/20_fashion_mnist/cnn/04_cnn_conv_svd.ipynb
"""

import copy

import torch
from torch import nn

from ..utils.modules import get_named_module, set_named_module
from .svd import coerce_rank, truncated_svd


def conv2d_weight_matrix(weight: torch.Tensor) -> torch.Tensor:
    """Conv2d 重みを SVD 用の 2 次元行列にする。

    ``(out_ch, in_ch, kH, kW)`` → ``(out_ch, in_ch * kH * kW)``。
    """
    return weight.flatten(start_dim=1)


def conv2d_max_rank(conv: nn.Conv2d) -> int:
    """現在の unfolding（out × in*kH*kW）に対する最大 rank。"""
    out_ch, in_ch, k_h, k_w = conv.weight.shape
    return min(out_ch, in_ch * k_h * k_w)


def _factory_kwargs(conv: nn.Conv2d) -> dict:
    return {
        "device": conv.weight.device,
        "dtype": conv.weight.dtype,
    }


def _unsupported_conv2d(conv: nn.Conv2d) -> None:
    """現在の flattening（out × in*kH*kW）が groups=1 前提のため、それ以外は拒否する。"""
    if conv.groups != 1:
        raise ValueError(
            "groups=1 の Conv2d のみ対応しています: "
            f"groups={conv.groups}"
        )
    if getattr(conv, "transposed", False):
        raise ValueError("転置畳み込みは未対応です。")


def factorize_conv2d_layer(conv: nn.Conv2d, rank: int) -> nn.Sequential:
    """Conv2d を SVD で低ランクな 2 層 Conv へ分解する。

    ``W`` は ``(out, in, kH, kW)``。平坦化して ``W_mat ≈ B @ A`` とする。

    - ``A = Vh_r``: 1 層目 ``Conv(in → rank, 元 kernel)``。中間 rank チャネル
    - ``B = U_r @ diag(S_r)``: 2 層目 ``1×1 Conv(rank → out)``。空間サイズは不変

    stride / padding / dilation / padding_mode は 1 層目が引き継ぐ。
    bias は最終出力に加わるため 2 層目へ移す。
    生成層は元層と同じ device / dtype で作り、float32 を経由しない。
    ``requires_grad`` も元層から引き継ぐ。
    """
    _unsupported_conv2d(conv)
    rank = coerce_rank(rank, conv2d_max_rank(conv))
    factory = _factory_kwargs(conv)

    out_ch, in_ch, k_h, k_w = conv.weight.shape
    W_mat = conv2d_weight_matrix(conv.weight.detach())
    U_r, S_r, Vh_r = truncated_svd(W_mat, rank)

    A = Vh_r
    B = U_r @ torch.diag(S_r)

    first_layer = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=rank,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=False,
        padding_mode=conv.padding_mode,
        **factory,
    )
    second_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=conv.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=(conv.bias is not None),
        padding_mode="zeros",
        **factory,
    )

    with torch.no_grad():
        first_layer.weight.copy_(A.reshape(rank, in_ch, k_h, k_w))
        second_layer.weight.copy_(B.reshape(out_ch, rank, 1, 1))
        if conv.bias is not None:
            second_layer.bias.copy_(conv.bias)

    first_layer.weight.requires_grad = conv.weight.requires_grad
    second_layer.weight.requires_grad = conv.weight.requires_grad
    if conv.bias is not None:
        second_layer.bias.requires_grad = conv.bias.requires_grad

    return nn.Sequential(first_layer, second_layer)


def factorize_named_conv2d(
    model: nn.Module,
    layer_name: str,
    rank: int,
) -> nn.Module:
    """指定した Conv2d だけを低ランク 2 層へ置換したコピーを返す。

    ``layer_name`` は ``\"conv2\"`` でも ``\"block.conv\"`` でもよい。
    元の ``model`` は変更しない。``deepcopy`` するため Fine-tune しても
    baseline の Parameter は動かない。
    """
    compressed_model = copy.deepcopy(model)
    layer = get_named_module(model, layer_name)
    if not isinstance(layer, nn.Conv2d):
        raise TypeError(
            f"{layer_name!r} は Conv2d である必要があります: {type(layer).__name__}"
        )
    set_named_module(
        compressed_model,
        layer_name,
        factorize_conv2d_layer(layer, rank),
    )
    return compressed_model


# 既存 Notebook 向けの旧名
factorize_Conv2d_layer = factorize_conv2d_layer
