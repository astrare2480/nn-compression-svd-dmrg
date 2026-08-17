"""MNISTMLP / Fashion-MNIST MLP にSVD圧縮を適用するモデル組み立て。

移植元:
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
    「2層に分解し、SVDを適用したモデルで評価」のセル

SVD そのものは ``compression.svd``、Linear の因子分解は
``compression.linear_svd`` に置く。このファイルには fc1・fc2・fc3
という現在の MLP 構造を前提にした組み立てだけを置く。
"""

from ..models.mlp import MNISTMLP
from .linear_svd import factorize_linear_layer, rebuild_linear_from_svd
from .svd import truncated_svd


def make_one_layer_svd_model(model, fc1_rank, fc2_rank):
    """fc1・fc2 を低ランク近似した1層Linearへ差し替えた MLP を作る.

    層の形状は変わらないため、パラメータ削減ではなく近似誤差を確認する
    実験用である。fc3 は小さい分類層なので元の層を共有する。
    """
    device = model.fc1.weight.device
    compressed_model = MNISTMLP().to(device)
    compressed_model.fc1 = rebuild_linear_from_svd(
        *truncated_svd(model.fc1.weight.detach(), fc1_rank),
        model.fc1,
    )
    compressed_model.fc2 = rebuild_linear_from_svd(
        *truncated_svd(model.fc2.weight.detach(), fc2_rank),
        model.fc2,
    )
    compressed_model.fc3 = model.fc3
    return compressed_model


def make_two_layer_svd_model(model, fc1_rank, fc2_rank):
    """fc1・fc2を指定rankの2層Linearへ置換した MLPを作る.

    ``Linear(in -> out)`` を ``Linear(in -> rank)`` と
    ``Linear(rank -> out)`` に分解するため、rank が十分小さい場合に
    パラメータ数・理論MACsを削減できる。
    """
    device = model.fc1.weight.device
    compressed_model = MNISTMLP().to(device)
    compressed_model.fc1 = factorize_linear_layer(model.fc1, fc1_rank)
    compressed_model.fc2 = factorize_linear_layer(model.fc2, fc2_rank)
    compressed_model.fc3 = model.fc3
    return compressed_model
