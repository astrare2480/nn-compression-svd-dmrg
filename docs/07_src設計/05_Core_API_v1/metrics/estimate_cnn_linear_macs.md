# `estimate_cnn_linear_macs`

**Stability:** B  
**定義:** `src/nn_compression/metrics/cnn_macs.py`

## 責務

Fashion-MNIST CNNのLinear部について、fc1だけを低rank2層化したMACsを比較する。

## Signature

```python
estimate_cnn_linear_macs(
    model,
    fc1_rank,
    verbose=True,
)
```

## 引数

CNN model、fc1 rank、表示有無。

## 戻り値

`(baseline_macs, compressed_macs, compute_reduction_rate)`。

## 使用場面

Fashion-MNIST CNNのLinear-only SVD実験。

## ざっくりした処理

baselineはfc1+fc2、compressedはfactorized fc1+original fc2として集計する。

## 主なcontract / 注意事項

現行CNNの`fc1/fc2`名を前提とするexperiment-support API。

## 関連API

`compressed_linear_macs`, `estimate_cnn_macs`
