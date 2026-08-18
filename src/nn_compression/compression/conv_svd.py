"""nn.Conv2d に対する汎用 SVD 処理。

参照元:
    notebooks/20_fashion_mnist/cnn/04_cnn_conv_svd.ipynb
"""

import copy

import torch
from torch import nn

from ..utils.modules import get_named_module, set_named_module
from .svd import truncated_svd


def conv2d_weight_matrix(weight: torch.Tensor) -> torch.Tensor:
    """Conv2d 重みを SVD 用の 2 次元行列にする。

    ``(out_ch, in_ch, kH, kW)`` → ``(out_ch, in_ch * kH * kW)``。
    """
    return weight.flatten(start_dim=1)


def factorize_conv2d_layer(conv: nn.Conv2d, rank: int) -> nn.Sequential:
    """Conv2d を SVD で低ランクな 2 層 Conv へ分解する。

    ``W`` は ``(out, in, kH, kW)``。平坦化して ``W_mat ≈ B @ A`` とする。

    - ``A = Vh_r``: 1 層目 ``Conv(in → rank, 元 kernel)``。中間 rank チャネル
    - ``B = U_r @ diag(S_r)``: 2 層目 ``1×1 Conv(rank → out)``。空間サイズは不変

    Conv 重みは出力チャネルが先頭次元なので、B の各行が
    「1 つの最終チャネルを作る混合係数」になる。
    stride / padding / dilation は 1 層目が引き継ぐ。
    bias は最終出力に加わるため 2 層目へ移す。``groups != 1`` は非対応。
    """
    if conv.groups != 1:
        raise ValueError("groups=1 のConv2dのみ対応しています。")

    out_ch, in_ch, k_h, k_w = conv.weight.shape
    max_rank = min(out_ch, in_ch * k_h * k_w)
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank は1〜{max_rank}の範囲で指定してください。")

    W_mat = conv2d_weight_matrix(conv.weight.detach())
    U_r, S_r, Vh_r = truncated_svd(W_mat, rank)

    A = Vh_r
    B = U_r @ torch.diag(S_r)

    # A を (rank, in, kH, kW) に戻して通常の畳み込みへ使う。
    first_layer = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=rank,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=False,
    )
    # 1×1 / stride=1 / padding=0 で空間サイズは変えない。
    second_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=conv.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=(conv.bias is not None),
    )

    with torch.no_grad():
        first_layer.weight.copy_(A.reshape(rank, in_ch, k_h, k_w))
        second_layer.weight.copy_(B.reshape(out_ch, rank, 1, 1))
        if conv.bias is not None:
            second_layer.bias.copy_(conv.bias)

    return nn.Sequential(first_layer, second_layer).to(
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )


def factorize_named_conv2d(
    model: nn.Module,
    layer_name: str,
    rank: int,
) -> nn.Module:
    """指定した Conv2d だけを低ランク 2 層へ置換したコピーを返す。

    ``layer_name`` は ``\"conv2\"`` でも ``\"block.conv\"`` でもよい。
    元の ``model`` は変更しない。
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
