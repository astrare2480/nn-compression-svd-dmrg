# `CIFAR10CNN`

**Stability:** B  
**定義:** `src/nn_compression/models/cifar10.py`

## 責務

CIFAR-10 SVD/Tucker実験用のConv×3 + GAP + 2 Linear baseline architectureを提供する。

## Signature

```python
CIFAR10CNN(
    *,
    in_channels: int = 3,
    conv_channels: tuple[int, int, int] = (32, 64, 128),
    hidden_dim: int = 256,
    num_classes: int = 10,
    dropout: float = 0.5,
)
```

## 引数

入力channel、3段Conv channel数、hidden dimension、class数、dropout率。

## 戻り値

`nn.Module` instance。

## 使用場面

CIFAR-10 baseline、複数Conv SVD、Tucker-2/HOOI比較。

## ざっくりした処理

```text
[Conv→ReLU→Pool] ×3
→ AdaptiveAvgPool 1x1
→ Flatten
→ Linear→ReLU→Dropout
→ Linear
```

## 主なcontract / 注意事項

既存checkpoint/resultsが依存するdefault主要layer名を維持する。

```text
conv1, conv2, conv3, fc1, fc2
```

## 関連API

`factorize_named_layers`, `build_tucker2_conv`, `estimate_cnn_macs`
