"""NN層の ``weight`` から特異値エネルギーを読む。

汎用の SVD（``truncated_svd``、整数 rank、行列化）は ``nn_compression.linalg``。
Linear / Conv2d への因子分解は ``linear_svd`` / ``conv_svd`` に置く。
"""

from __future__ import annotations

from torch import nn

from ..linalg import svd as _linalg_svd


def retained_energy(layer: nn.Module, rank: int) -> float:
    """層の ``weight`` から上位特異値エネルギーの割合を返す。

    Linear も Conv2d も ``layer.weight`` を 2 次元化してから計算する。
    """
    return _linalg_svd.retained_energy_from_matrix(
        _linalg_svd.as_matrix(layer.weight.detach()),
        rank,
    )


# 既存 Notebook / mlp_svd 向けの旧名。実装は ``linalg.truncated_svd``。
SVD = _linalg_svd.truncated_svd


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
