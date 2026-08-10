"""MNIST / Fashion-MNIST 実験で共通利用する MLP。

参照元:
    notebooks/20_fashion_mnist/mlp/00_mlp_baseline_fixed_50epochs.ipynb
    （MNISTMLP 定義セル）
"""

import torch.nn as nn


class MNISTMLP(nn.Module):
    """28×28グレースケール画像用の 784 -> 512 -> 256 -> 10 MLP.

    MNIST と Fashion-MNIST はともに入力形状・クラス数が同じため、
    元Notebookの ``MNISTMLP`` 名を維持して両方で再利用する。
    fc1 / fc2 は SVD 圧縮の対象、fc3 は小さい分類層として残す。
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """画像を平坦化して、3つの全結合層で logits を返す。"""
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x
