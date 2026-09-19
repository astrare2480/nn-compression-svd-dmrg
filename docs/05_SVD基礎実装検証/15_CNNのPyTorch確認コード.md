# 15 CNNのPyTorch確認コード

数式と意味は各例にリンクした基礎理論ノートを参照する。ここにはPyTorchで確認するコードだけを置く。複数の例は独立実行を前提とせず、各例の前提条件を理論ノートで確認する。

---

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/02_ニューラルネットワーク基礎/15_CNNとConv2dの基礎]]（対応コード）

```python
from nn_compression.models import FashionMNISTCNN
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/02_ニューラルネットワーク基礎/15_CNNとConv2dの基礎]]（7. Poolingも空間サイズを変える）

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/02_ニューラルネットワーク基礎/15_CNNとConv2dの基礎]]（10. kernel size と patch size）

```python
kernel_size=patch_size
stride=patch_size
```

