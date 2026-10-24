# `ClassificationMetrics`

**Stability:** A
**定義:** `src/nn_compression/metrics/classification_outputs.py`

## 責務

計算済みlogitsから得たloss、accuracy、sample数をDataFrame化しやすいPython scalarで保持する不変dataclass。

## Signature
```python
ClassificationMetrics(loss, accuracy, num_samples)
```

## 引数

- `loss`: 全sample平均cross entropy。
- `accuracy`: `argmax`正解率。
- `num_samples`: 評価sample数。

## 戻り値

3 fieldを持つfrozen instance。

## 使用場面

rank候補ごとの分類性能をrecordへ格納するとき。

## 処理概要

値の計算は`classification_metrics_from_logits`が担当する。

## 主なcontract / 注意事項

- `dataclasses.asdict`で1行recordへ変換できる。
- `accuracy`の想定範囲は`[0,1]`。

## 関連API

`classification_metrics_from_logits`
