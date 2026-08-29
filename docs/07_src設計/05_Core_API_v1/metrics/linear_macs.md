# `linear_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

1 sampleをLinearへ通す理論MACsを計算する。

## Signature

```python
linear_macs(layer) -> int
```

## 引数

`layer`: `in_features/out_features`を持つLinear。

## 戻り値

```text
in_features * out_features
```

## 使用場面

baseline Linearの理論計算量を数えるとき。

## ざっくりした処理

Linearの入力dimensionと出力dimensionを掛ける。

## 主なcontract / 注意事項

bias加算やactivation等は含めない理論MACs。

## 関連API

`compressed_linear_macs`, `estimate_mlp_macs`, `estimate_cnn_macs`
