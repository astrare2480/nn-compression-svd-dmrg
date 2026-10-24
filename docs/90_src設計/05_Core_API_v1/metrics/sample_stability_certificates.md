# `sample_stability_certificates`

**Stability:** A
**定義:** `src/nn_compression/metrics/prediction_stability.py`

## 責務

sampleごとにlogits誤差上界がclassification marginの半分未満かを判定する。

## Signature
```python
sample_stability_certificates(margins, error_bound)
```

## 引数

- `margins`: shape `(batch,)`の有限な非負margin。
- `error_bound`: Python実数、0階Tensor、またはshape `(batch,)`の非負上界。Tensorは同じdevice、sample別Tensorは同じdtypeを要求する。

## 戻り値

shape `(batch,)`のbool Tensor。要素$n$は$2U_n<\gamma_n$の真偽。

## 使用場面

sample別logits上界から、圧縮後もargmaxが変わらないことを保証するとき。

## 処理概要

Python実数はfloat64、0階Tensorはmarginとのpromote dtype、sample別Tensorは同dtypeで比較する。除算を避け`2*bound < margin`を厳密不等号で評価する。

## 主なcontract / 注意事項

- 境界$U_n=\gamma_n/2$は不成立。margin 0も成立しない。
- 上界を狭いdtypeへ丸めないため、微小な正値を0にしてfalse certificateを作らない。
- 負、NaN、infの上界を拒否する。

## 関連API

`classification_margins`, `batch_stability_certificate`
