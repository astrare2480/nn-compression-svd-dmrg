# `accuracy_drop`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline accuracyからcompressed accuracyがどれだけ低下したか計算する。

## Signature

```python
accuracy_drop(
    baseline_accuracy,
    compressed_accuracy,
    verbose=True,
) -> float
```

## 引数

baseline/compressed accuracyと表示有無。

## 戻り値

`baseline_accuracy - compressed_accuracy`。

## 使用場面

圧縮前後のtask性能差を簡潔に記録するとき。

## ざっくりした処理

2つのaccuracyを引き、必要なら表示する。

## 主なcontract / 注意事項

正なら精度低下、負ならcompressed側のaccuracyが高い。

## 関連API

`collect_compression_metrics`
