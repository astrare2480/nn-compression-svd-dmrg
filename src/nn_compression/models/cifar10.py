"""小さな画像分類 CNN。CIFAR-10 実験の baseline から一般化。

参照元:
    notebooks/30_cifar10/02_svd_global_compression.ipynb
"""

import torch
from torch import nn


class CIFAR10CNN(nn.Module):
    """Conv×3 + GAP + 2層 Linear の分類 CNN。

    既定値は CIFAR-10 実験と同じ（3×32×32、10クラス）。
    チャネル数・クラス数は引数で変える。
    MaxPool2d(kernel_size=2) を 3 回通すため、入力の高さ・幅は
    各々 8 以上の偶数を想定する。GAP は最後の空間サイズを 1×1 にする。
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        c1, c2, c3 = conv_channels

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(c3, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """画像 batch からクラス logits を返す。"""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
