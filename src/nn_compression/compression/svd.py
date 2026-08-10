"""Linear層に対する汎用的なSVD分解・低ランク近似。

参照元:
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/10_mnist_mlp/02_rank_accuracy_tradeoff.ipynb

元NotebookのSVD演算とLinear層の因子分解を切り出したもの。
現在のMLPの ``fc1`` / ``fc2`` を差し替える組み立ては、モデル構造に
依存するため ``compression.mlp_svd`` に分離している。
"""

import torch
import torch.nn as nn

def SVD(W, r):
    """重み行列 ``W`` の上位 ``r`` 成分だけを残した SVD を返す。

    ``W ≈ U_r @ diag(S_r) @ Vh_r`` と近似する。rank が小さいほど
    パラメータ数と演算量は減る一方、元の重みとの差は大きくなる。
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    return U_r, S_r, Vh_r


def RebuildSVD(U_r, S_r, Vh_r, layer):
    """SVD近似から、元層と同じ形状の1つの Linear 層を再構成する。

    元のバイアスはそのまま引き継ぐ。層数は変わらないため、これは
    パラメータ削減ではなく、低ランク近似の誤差確認向けである。
    """
    W_r = U_r @ torch.diag(S_r) @ Vh_r

    new_layer = nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=True,
    )

    with torch.no_grad():
        new_layer.weight.copy_(W_r)
        new_layer.bias.copy_(layer.bias)

    return new_layer.to(layer.weight.device)


def factorize_linear_layer(layer, rank):
    """Linear層を rank 次元の2層 Linear に置き換える。

    元の重み ``W`` を ``U_r @ diag(S_r) @ Vh_r`` と近似し、
    1層目へ ``Vh_r``、2層目へ ``U_r @ diag(S_r)`` を代入する。
    バイアスは2層目だけが保持する。これは元Notebookの
    ``devidetwolayer``（スペル修正）の処理に対応する。
    """
    U_r, S_r, Vh_r = SVD(layer.weight.detach(), rank)
    first_layer = nn.Linear(layer.in_features, rank, bias=False)
    second_layer = nn.Linear(rank, layer.out_features, bias=True)

    with torch.no_grad():
        first_layer.weight.copy_(Vh_r)
        second_layer.weight.copy_(U_r @ torch.diag(S_r))
        second_layer.bias.copy_(layer.bias.data)

    return nn.Sequential(first_layer, second_layer).to(layer.weight.device)


def retained_energy(layer, rank):
    """上位特異値が重みの Frobenius ノルム二乗を保つ割合を返す。

    ``sum(S[:rank]**2) / sum(S**2)``。1に近いほど重み近似の情報を
    多く残している目安だが、分類精度そのものを保証する値ではない。
    """
    singular_values = torch.linalg.svdvals(
        layer.weight.detach()
    )

    return (
        singular_values[:rank].square().sum()
        / singular_values.square().sum()
    ).item()
