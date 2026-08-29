# `logits_rmse`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline/compressed modelのlogits全要素のRMSEを計算する。

## Signature

```python
logits_rmse(
    baseline_model,
    compressed_model,
    loader,
    device,
) -> float
```

## 引数

2モデル、loader、device。

## 戻り値

logits差のroot mean squared error。

## 使用場面

argmax一致率より細かく、出力分布のずれを測りたいとき。

## ざっくりした処理

```text
両modelをeval
→ 各batchのlogits差を二乗
→ 全要素へ合計
→ element_countで平均
→ sqrt
```

## 主なcontract / 注意事項

empty loader拒否、両modelのtraining状態を復元。

## 関連API

`agreement`, `collect_compression_metrics`
