# `MinimumMargin`

**Stability:** A
**定義:** `src/nn_compression/metrics/prediction_stability.py`

## 責務

batch内の最小classification marginと、そのbottleneck sample indexを保持する不変dataclass。

## Signature
```python
MinimumMargin(value, sample_index)
```

## 引数

- `value`: 最小marginを表す0階Tensor。
- `sample_index`: 最初に最小値を取ったsampleの0階`torch.int64` index。

## 戻り値

`value`と`sample_index`を持つfrozen instance。

## 使用場面

batch-wide certificateのボトルネックを特定するとき。

## 処理概要

値の計算は`minimum_classification_margin`が担当し、このclassは結果を名前付きで保持する。

## 主なcontract / 注意事項

- 同率最小値では先頭sampleを表す。
- `value`のdtype/deviceは入力marginを維持する。

## 関連API

`minimum_classification_margin`, `BatchStabilityCertificate`
