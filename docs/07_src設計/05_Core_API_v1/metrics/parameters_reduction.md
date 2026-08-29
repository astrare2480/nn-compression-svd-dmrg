# `parameters_reduction`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baselineに対するcompressed modelのParameter削減率を計算する。

## Signature

```python
parameters_reduction(
    baseline_model,
    compressed_model,
) -> float
```

## 引数

baseline modelとcompressed model。

## 戻り値

```text
1 - compressed_parameters / baseline_parameters
```

## 使用場面

model-levelの圧縮率評価。

## ざっくりした処理

両modelへ`count_parameters()`を適用して比率を計算する。

## 主なcontract / 注意事項

0.9なら90%削減を意味する。

## 関連API

`count_parameters`, `collect_compression_metrics`
