"""層種別・モデル構造に依存しない SVD の共通処理。

参照元:
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/cnn/04_cnn_conv_svd.ipynb

Linear / Conv2d への因子分解は ``linear_svd`` / ``conv_svd`` に置く。
"""

from __future__ import annotations

import operator

import torch
from torch import nn


def coerce_rank(rank, max_rank: int, *, name: str = "rank") -> int:
    """``rank`` を整数化し、``1 <= rank <= max_rank`` を検証する。"""
    try:
        rank_int = operator.index(rank)
    except TypeError as exc:
        raise TypeError(
            f"{name} は整数である必要があります: {rank!r}"
        ) from exc

    if max_rank < 1:
        raise ValueError(f"{name} の上限 {max_rank} が 1 未満です。")
    if not 1 <= rank_int <= max_rank:
        raise ValueError(
            f"{name} は 1〜{max_rank} の範囲で指定してください: {rank_int}"
        )
    return rank_int


def matrix_max_rank(matrix: torch.Tensor) -> int:
    """2 次元行列に対する SVD の最大 rank。"""
    if matrix.ndim != 2:
        raise ValueError(
            f"SVD には 2 次元行列が必要です: ndim={matrix.ndim}"
        )
    return int(min(matrix.shape))


def truncated_svd(matrix: torch.Tensor, rank: int):
    """2次元行列 ``matrix`` の上位 ``rank`` 成分だけを残した SVD を返す。

    ``matrix ≈ U_r @ diag(S_r) @ Vh_r``。形状は
    ``U_r: (out, rank)``、``S_r: (rank,)``、``Vh_r: (rank, in)``。
    入力の dtype / device を維持する。
    """
    rank = coerce_rank(rank, matrix_max_rank(matrix))
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    return U_r, S_r, Vh_r


def as_matrix(weight: torch.Tensor) -> torch.Tensor:
    """重み Tensor を SVD 用の 2 次元行列 ``(out, -1)`` にする。

    Linear はすでに 2 次元。Conv2d は
    ``(out_channels, in_channels, kH, kW)`` を
    ``(out_channels, in_channels * kH * kW)`` へ平坦化する。
    """
    if weight.ndim == 2:
        return weight
    if weight.ndim > 2:
        return weight.reshape(weight.shape[0], -1)
    raise ValueError(f"2次元以上の重みが必要です: ndim={weight.ndim}")


def retained_energy_from_matrix(matrix: torch.Tensor, rank: int) -> float:
    """2次元行列に対する上位特異値エネルギーの割合を返す。

    ``sum(S[:rank]**2) / sum(S**2)``。1 に近いほど近似が元の行列の
    Frobenius ノルムを多く残している目安だが、分類精度を保証しない。
    """
    rank = coerce_rank(rank, matrix_max_rank(matrix))
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square().sum()
    if energy.item() == 0.0:
        return 1.0
    return (singular_values[:rank].square().sum() / energy).item()


def retained_energy(layer: nn.Module, rank: int) -> float:
    """層の ``weight`` から上位特異値エネルギーの割合を返す。

    Linear も Conv2d も ``layer.weight`` を 2 次元化してから計算する。
    """
    return retained_energy_from_matrix(
        as_matrix(layer.weight.detach()),
        rank,
    )


# 既存 Notebook / mlp_svd 向けの旧名
SVD = truncated_svd


def __getattr__(name: str):
    """旧 ``svd.py`` から Linear 因子分解を import していたコード向け。"""
    if name in {
        "RebuildSVD",
        "factorize_linear_layer",
        "rebuild_linear_from_svd",
    }:
        from . import linear_svd

        return getattr(linear_svd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
