# `MNISTMLP`

**Stability:** B  
**定義:** `src/nn_compression/models/mlp.py`

## 責務

MNIST/Fashion-MNISTのMLP SVD実験で共通利用するbaseline architectureを提供する。

## Signature

```python
MNISTMLP()
```

## 引数

なし。

## 戻り値

`nn.Module` instance。

## 使用場面

MNIST/Fashion-MNIST MLP baseline、Linear SVD、Fine-tuning。

## ざっくりした処理

```text
inputをflatten
→ Linear 784→512
→ ReLU
→ Linear 512→256
→ ReLU
→ Linear 256→10
→ logits
```

## 主なcontract / 注意事項

Core API v1中は主要layer名を安定化する。

```text
fc1, relu1, fc2, relu2, fc3
```

## 関連API

`make_one_layer_svd_model`, `make_two_layer_svd_model`, `estimate_mlp_macs`
