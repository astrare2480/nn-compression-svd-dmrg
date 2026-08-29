# `benchmark_inference_print`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline/compressed modelを同一input条件で連続benchmarkし、必要なら結果を表示する。

## Signature

```python
benchmark_inference_print(
    baseline_model,
    compressed_model,
    data_loader,
    device,
    verbose=True,
    *,
    warmup=10,
    repeats=5000,
    input_batch=None,
)
```

## 引数

2モデル、batch取得元、device、benchmark条件。

## 戻り値

`(baseline_time_s, compressed_time_s)`。

## 使用場面

Notebookで圧縮前後latencyを簡単に比較・表示するとき。

## ざっくりした処理

```text
必要なら共通input_batchを1回取得
→ baseline benchmark
→ compressed benchmark
→ verboseなら条件/時間表示
```

## 主なcontract / 注意事項

両modelのtraining状態を処理前後で復元する。

## 関連API

`benchmark_inference`, `take_inference_batch`
