# `classification_margins`

**Stability:** A
**定義:** `src/nn_compression/metrics/prediction_stability.py`

## 責務

各sampleのtop-1 logitとtop-2 logitの差をclassification marginとして計算する。

## Signature
```python
classification_margins(logits)
```

## 引数

- `logits`: shape `(batch, num_classes)`、`batch>=1`, `num_classes>=2`の有限float32/float64 Tensor。

## 戻り値

shape `(batch,)`の非負Tensor。sample $n$について

$$
\gamma_n=l_{n,(1)}-l_{n,(2)}
$$

を表す。tieなら0。

## 使用場面

argmax不変条件$U_n<\gamma_n/2$を評価するとき。

## 処理概要

`torch.topk(..., k=2)`で上位2値を取り、有限な下界となるよう差を計算する。

## 主なcontract / 注意事項

- 極端な有限値の減算がoverflowすると最大有限値へcapする。
- 上向き丸めは1 ulp下げ、certificateを危険側へずらさない。
- capまたは1 ulp補正された行の勾配は0。通常行は`top-second`の勾配を保つ。

## 関連API

`minimum_classification_margin`, `sample_stability_certificates`
