# `collect_compression_metrics`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

1つのcompressed candidateについて、validation・parameter・latency・出力差などの共通指標を1 recordへまとめる。

## Signature

```python
collect_compression_metrics(
    baseline_model,
    compressed_model,
    loader,
    criterion,
    device,
    *,
    baseline_acc: float,
    baseline_time_s: float,
    warmup: int = 5,
    repeats: int = 200,
    compressed_macs: int | None = None,
    compute_reduction: float | None = None,
    baseline_macs: int | None = None,
    include_model: bool = False,
    input_batch=None,
) -> dict
```

## 引数

baseline/compressed model、評価loader、criterion/device、baseline指標、benchmark条件、optional MACs、model保持有無。

## 戻り値

parameters、validation loss/acc、accuracy drop、latency条件、agreement、logits RMSE、optional MACs/modelを含むdict。

## 使用場面

rank sweepや候補比較表の共通record生成。

## ざっくりした処理

```text
evaluate(compressed)
→ input batch確保
→ benchmark_inference
→ parameter metrics
→ agreement
→ logits_rmse
→ optional MACs/modelを追加
```

## 主なcontract / 注意事項

- `include_model=False`が既定。多数candidateのmodelを表へ保持しない。
- 同じloaderを複数回走査するため**re-iterable loader前提**。one-shot iteratorは未対応。
- baseline/compressedのtraining状態を復元。

## 関連API

`evaluate`, `benchmark_inference`, `agreement`, `logits_rmse`, `sweep_layer_ranks`
