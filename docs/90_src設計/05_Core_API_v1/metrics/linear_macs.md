# `linear_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

1 sampleを`Linear(in_features → out_features)`へ通すときの理論MACs（積和回数）を計算する。

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

## 処理概要

1. **入力特徴数`in_features`を読む。**
2. **出力特徴数`out_features`を読む。**
3. **各出力要素が入力dimension分の積和を行うとみなし、2つを掛ける。**
4. **1 sampleあたりの理論MACsとして返す。**

## 主なcontract / 注意事項

bias加算、activation、memory access等は含めない単純な理論MACs。

## 関連API

`compressed_linear_macs`, `estimate_mlp_macs`, `estimate_cnn_macs`
