"""nn.Linear に対する汎用 SVD 処理。

参照元:
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/cnn/02_cnn_linear_svd.ipynb
"""

import copy

import torch
from torch import nn

from ..utils.modules import get_named_module, set_named_module
from .svd import truncated_svd


def rebuild_linear_from_svd(U_r, S_r, Vh_r, layer: nn.Linear) -> nn.Linear:
    """SVD 近似から、元層と同じ形状の 1 つの Linear 層を再構成する。

    元に bias がある場合だけ引き継ぐ。層数は変わらないため、これは
    パラメータ削減ではなく、低ランク近似の誤差確認向けである。
    """
    W_r = U_r @ torch.diag(S_r) @ Vh_r
    has_bias = layer.bias is not None

    new_layer = nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=has_bias,
    )

    with torch.no_grad():
        new_layer.weight.copy_(W_r)
        if has_bias:
            new_layer.bias.copy_(layer.bias)

    return new_layer.to(
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )


def factorize_linear_layer(layer: nn.Linear, rank: int) -> nn.Sequential:
    """Linear 層を rank 次元の 2 層 Linear に置き換える。

    元の重み ``W`` を ``U_r @ diag(S_r) @ Vh_r`` と近似し、
    1 層目へ ``Vh_r``、2 層目へ ``U_r @ diag(S_r)`` を代入する。
    元に bias がある場合だけ、2 層目へコピーする。
    """
    U_r, S_r, Vh_r = truncated_svd(layer.weight.detach(), rank)
    has_bias = layer.bias is not None
    first_layer = nn.Linear(layer.in_features, rank, bias=False)
    second_layer = nn.Linear(rank, layer.out_features, bias=has_bias)

    with torch.no_grad():
        first_layer.weight.copy_(Vh_r)
        second_layer.weight.copy_(U_r @ torch.diag(S_r))
        if has_bias:
            second_layer.bias.copy_(layer.bias)

    return nn.Sequential(first_layer, second_layer).to(
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )


def factorize_named_linear(
    model: nn.Module,
    layer_name: str,
    rank: int,
) -> nn.Module:
    """指定した Linear だけを低ランク 2 層へ置換したコピーを返す。

    ``Linear(in → out)`` を ``Linear(in → rank)`` + ``Linear(rank → out)``
    に分解する。``layer_name`` は ``\"fc1\"`` でも ``\"head.fc\"`` でもよい。
    元の ``model`` は変更しない。
    """
    compressed_model = copy.deepcopy(model)
    layer = get_named_module(model, layer_name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(
            f"{layer_name!r} は Linear である必要があります: {type(layer).__name__}"
        )
    set_named_module(
        compressed_model,
        layer_name,
        factorize_linear_layer(layer, rank),
    )
    return compressed_model


# 既存コード向けの旧名
RebuildSVD = rebuild_linear_from_svd
