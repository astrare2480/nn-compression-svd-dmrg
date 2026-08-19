"""MNISTMLP / Fashion-MNIST MLP にSVD圧縮を適用するモデル組み立て。

移植元:
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
    「2層に分解し、SVDを適用したモデルで評価」のセル

SVD そのものは ``compression.svd``、Linear の因子分解は
``compression.linear_svd`` に置く。このファイルには fc1・fc2・fc3
という現在の MLP 構造を前提にした組み立てだけを置く。
"""

from __future__ import annotations

import copy

from .linear_svd import factorize_linear_layer, rebuild_linear_from_svd
from .svd import truncated_svd


def _resolve_mlp_ranks(fc1_rank, fc2_rank, r1, r2) -> tuple[int, int]:
    """primary 名と legacy 名 ``r1`` / ``r2`` を解決する。両方指定はエラー。"""
    if fc1_rank is not None and r1 is not None:
        raise ValueError("fc1_rank と r1 を同時に指定しないでください。")
    if fc2_rank is not None and r2 is not None:
        raise ValueError("fc2_rank と r2 を同時に指定しないでください。")

    resolved_fc1 = fc1_rank if fc1_rank is not None else r1
    resolved_fc2 = fc2_rank if fc2_rank is not None else r2
    if resolved_fc1 is None or resolved_fc2 is None:
        raise TypeError(
            "fc1_rank（または r1）と fc2_rank（または r2）が必要です。"
        )
    return resolved_fc1, resolved_fc2


def make_one_layer_svd_model(
    model,
    fc1_rank=None,
    fc2_rank=None,
    *,
    r1=None,
    r2=None,
):
    """fc1・fc2 を低ランク近似した1層Linearへ差し替えた MLP を作る.

    層の形状は変わらないため、パラメータ削減ではなく近似誤差を確認する
    実験用である。返り値は ``model`` と Parameter を共有しない（``deepcopy``。fc3 も含む）。
    ``r1`` / ``r2`` は旧 Notebook 向け alias。
    """
    fc1_rank, fc2_rank = _resolve_mlp_ranks(fc1_rank, fc2_rank, r1, r2)
    compressed_model = copy.deepcopy(model)
    compressed_model.fc1 = rebuild_linear_from_svd(
        *truncated_svd(model.fc1.weight.detach(), fc1_rank),
        model.fc1,
    )
    compressed_model.fc2 = rebuild_linear_from_svd(
        *truncated_svd(model.fc2.weight.detach(), fc2_rank),
        model.fc2,
    )
    return compressed_model


def make_two_layer_svd_model(
    model,
    fc1_rank=None,
    fc2_rank=None,
    *,
    r1=None,
    r2=None,
):
    """fc1・fc2を指定rankの2層Linearへ置換した MLPを作る.

    ``Linear(in -> out)`` を ``Linear(in -> rank)`` と
    ``Linear(rank -> out)`` に分解するため、rank が十分小さい場合に
    パラメータ数・理論MACsを削減できる。
    返り値は ``model`` と Parameter を共有しない（``deepcopy``。fc3 も含む）。
    ``r1`` / ``r2`` は旧 Notebook 向け alias。
    """
    fc1_rank, fc2_rank = _resolve_mlp_ranks(fc1_rank, fc2_rank, r1, r2)
    compressed_model = copy.deepcopy(model)
    compressed_model.fc1 = factorize_linear_layer(model.fc1, fc1_rank)
    compressed_model.fc2 = factorize_linear_layer(model.fc2, fc2_rank)
    return compressed_model
