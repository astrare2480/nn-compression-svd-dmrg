"""Fashion-MNIST 分類用 CNN。

参照元:
    notebooks/20_fashion_mnist/cnn/01_cnn_baseline.ipynb
"""

import torch
from torch import nn


class FashionMNISTCNN(nn.Module):
    """1×28×28 グレースケール画像用の小さな CNN。

    Conv(1→32) → Pool → Conv(32→64) → Pool → Linear(3136→128) → Linear(128→10)。
    圧縮実験では ``conv2`` と ``fc1`` を SVD 対象にする。

    Conv1d/2d/3d の違いは畳み込む空間方向の数で、チャネルは別次元。
    ``kernel=3, padding=1, stride=1`` で空間サイズは維持し、
    2×2 MaxPool（stride=2）で 28→14→7。Flatten 直前は 64×7×7=3136。
    ReLU が無いと畳み込みを重ねても線形のままになる。
    """

    def __init__(self) -> None:
        super().__init__()

        # in_channels=1 は Fashion-MNIST がグレースケールのため。
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.flatten = nn.Flatten()
        # 3136 は inspect_shapes() で確認した Flatten 後の特徴数。
        self.fc1 = nn.Linear(
            in_features=3136,
            out_features=128,
        )
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10,
        )

    def inspect_shapes(
        self,
        x: torch.Tensor | None = None,
    ) -> int:
        """各層の出力 shape を表示し、Flatten 後の特徴数を返す。

        Linear の ``in_features`` 確認用。引数なしならダミー ``(1, 1, 28, 28)``。
        ``train_loader`` は消費しない。実画像なら ``train_dataset[0]`` を
        ``unsqueeze(0)`` して ``(1, C, H, W)`` にして渡す。
        """
        if x is None:
            x = torch.zeros(1, 1, 28, 28)

        print("input:        ", tuple(x.shape))
        x = self.conv1(x)
        print("after conv1:  ", tuple(x.shape))
        x = self.relu1(x)
        x = self.pool1(x)
        print("after pool1:  ", tuple(x.shape))
        x = self.conv2(x)
        print("after conv2:  ", tuple(x.shape))
        x = self.relu2(x)
        x = self.pool2(x)
        print("after pool2:  ", tuple(x.shape))
        x = self.flatten(x)
        print("after flatten:", tuple(x.shape))

        flatten_dim = x.shape[1]
        print("flatten_dim:  ", flatten_dim)
        return flatten_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv → ReLU → Pool を 2 回したあと、Linear で 10 クラスの logits を返す。"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
