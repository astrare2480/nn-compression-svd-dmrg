# `BatchStabilityCertificate`

**Stability:** A
**定義:** `src/nn_compression/metrics/prediction_stability.py`

## 責務

batch全体のprediction stability判定に必要なmargin、bottleneck、sample別判定、全体判定を保持する不変dataclass。

## Signature
```python
BatchStabilityCertificate(margins, minimum_margin, bottleneck_index, sample_certified, all_certified)
```

## 引数

- `margins`: shape `(batch,)`のclassification margin。
- `minimum_margin`: batch最小marginの0階Tensor。
- `bottleneck_index`: 最小margin sampleのindex。
- `sample_certified`: shape `(batch,)`のbool判定。
- `all_certified`: 全sample成立を示す0階bool Tensor。

## 戻り値

上記5 fieldを持つfrozen instance。

## 使用場面

圧縮前logitsと誤差上界から、予測class不変を十分条件で保証するとき。

## 処理概要

値の生成は`batch_stability_certificate`が担当する。

## 主なcontract / 注意事項

- certificate不成立は予測変化を意味せず、「この上界では保証できない」ことを意味する。
- Tensorのdtype/deviceは生成元APIの契約に従う。

## 関連API

`batch_stability_certificate`, `sample_stability_certificates`
