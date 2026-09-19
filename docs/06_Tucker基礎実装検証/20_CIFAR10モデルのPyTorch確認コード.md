# 20 CIFAR10モデルのPyTorch確認コード

数式と意味は各例にリンクした基礎理論ノートを参照する。ここにはPyTorchで確認するコードだけを置く。複数の例は独立実行を前提とせず、各例の前提条件を理論ノートで確認する。

---

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]（PyTorchでは`AdaptiveAvgPool2d(1)`を使う）

```python
import torch
from torch import nn

# shape: (N, C, H, W) = (1, 1, 2, 2)
x = torch.tensor([
    [
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    ],
])

gap = nn.AdaptiveAvgPool2d(output_size=1)
pooled = gap(x)

print(pooled.shape)  # torch.Size([1, 1, 1, 1])
print(pooled)        # tensor([[[[2.5000]]]])

# batch軸を残し、C個のchannel値を1次元の特徴へ並べる。
features = torch.flatten(pooled, start_dim=1)
print(features.shape)  # torch.Size([1, 1])
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]（4. CIFAR-10モデルのshape）

```python
conv_channels = (32, 64, 128)
hidden_dim = 256
dropout = 0.5
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]（`Dropout`と`Dropout2d`の落とす単位）

```python
import torch
from torch import nn

torch.manual_seed(0)

# 1 sample、2 channel、各channelが2 x 2の特徴マップ。
x = torch.ones(1, 2, 2, 2)

dropout2d = nn.Dropout2d(p=0.5)
dropout2d.train()
y = dropout2d(x)

print(y.shape)  # torch.Size([1, 2, 2, 2])

# 各channelは全位置が0か、全位置が同じ倍率で残る。
for channel in range(y.shape[1]):
    channel_map = y[0, channel]
    assert (
        torch.all(channel_map == 0)
        or torch.all(channel_map != 0)
    )
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]（3チャネル重みを、全位置のCAMへ対応させる）

```python
import torch


def make_cam(
    feature_maps: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    """正のクラス寄与を残したCAMを返す。"""
    # 1 sampleを取り出し、チャネルKだけを縮約する。
    weights = classifier_weight[class_index]
    raw_cam = torch.einsum("k,khw->hw", weights, feature_maps[0])
    return torch.relu(raw_cam)


# 特徴マップとbiasは上の補足例。クラス番号0を例の猫クラスとする。
feature_maps = torch.tensor(
    [[[[1., 0.], [0., 1.]],
      [[0., 2.], [1., 0.]],
      [[1., 1.], [0., 2.]]]],
    dtype=torch.float64,
)
classifier_weight = torch.tensor([[1.2, 2.0, 0.8]], dtype=torch.float64)
cam = make_cam(feature_maps, classifier_weight, class_index=0)
expected = torch.tensor([[2.0, 4.8], [2.0, 2.8]], dtype=torch.float64)
assert torch.allclose(cam, expected)

# この例では全CAM要素が正なので、ReLU前後が同じになる。
pooled = feature_maps.mean(dim=(-2, -1))
score = (pooled @ classifier_weight.T)[0, 0] + 0.5
assert torch.allclose(score, cam.mean() + 0.5)
assert abs(score.item() - 3.4) < 1e-12
```

