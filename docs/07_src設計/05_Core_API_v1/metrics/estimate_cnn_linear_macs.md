# `estimate_cnn_linear_macs`

**Stability:** B  
**定義:** `src/nn_compression/metrics/cnn_macs.py`

## 責務

Fashion-MNIST CNNのLinear部について、`fc1`だけをSVD 2層化した場合の理論MACsを比較する。

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

## 処理概要

1. **baseline Linear部を`fc1 + fc2`として計算する。**
2. **compressed側の`fc1`を`in → rank → out`の2層式で計算する。**
3. **`fc2`は未圧縮なのでbaselineと同じMACsを加える。**
4. **全体削減率を計算する。**
5. **必要なら値を表示し、3要素tupleを返す。**

## 主なcontract / 注意事項

現行CNNの`fc1/fc2`属性名を前提とするexperiment-support API。

## 関連API

`compressed_linear_macs`, `estimate_cnn_macs`
