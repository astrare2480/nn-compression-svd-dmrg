# `conv2d_weight_matrix`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_svd.py`

## 責務

Conv2d weightをSVD用2次元行列へ平坦化する。

## Signature

```python
conv2d_weight_matrix(weight: torch.Tensor) -> torch.Tensor
```

## 引数

`weight`: 通常 `(C_out, C_in, kH, kW)` のConv2d weight。

## 戻り値

```text
(C_out, C_in * kH * kW)
```

のTensor。

## 使用場面

Conv SVDの前処理、Conv weightの行列rankを考えるとき。

## ざっくりした処理

`weight.flatten(start_dim=1)` で出力channel以外を1軸へまとめる。

## 主なcontract / 注意事項

この関数自体はgroups等のConv semanticsを検証しない。Module分解時は`factorize_conv2d_layer()`側で検証する。

## 関連API

`factorize_conv2d_layer`, `truncated_svd`
