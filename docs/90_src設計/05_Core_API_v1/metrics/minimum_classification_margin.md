# `minimum_classification_margin`

**Stability:** A
**定義:** `src/nn_compression/metrics/prediction_stability.py`

## 責務

sample別marginからbatch最小値とbottleneck indexを取得する。

## Signature
```python
minimum_classification_margin(margins)
```

## 引数

- `margins`: shape `(batch,)`の有限float32/float64 Tensor。空不可。

## 戻り値

`MinimumMargin`。`value`は入力と同じdtype/device、`sample_index`は0階int64 Tensor。

## 使用場面

batch全体を支配する最小margin sampleを診断するとき。

## 処理概要

`torch.argmin`を使い、同率なら先頭indexを選ぶ。

## 主なcontract / 注意事項

- NaN/inf、非1階、空入力を拒否する。
- 入力を変更せず、最小値へのautograd経路を保つ。

## 関連API

`MinimumMargin`, `classification_margins`, `batch_stability_certificate`
