# `relative_l2_from_squared_sums`

**Stability:** A
**定義:** `src/nn_compression/metrics/tensor_approximation.py`

## 責務

streamingで積んだ二乗和から、データ全体のrelative L2を返す。

## Signature

```python
relative_l2_from_squared_sums(
    difference_squared_sum: float,
    baseline_squared_sum: float,
) -> float
```

## 引数

- `difference_squared_sum`: `sum_i ||candidate_i - baseline_i||_2^2`。
- `baseline_squared_sum`: `sum_i ||baseline_i||_2^2`。

## 戻り値

`sqrt(difference_squared_sum) / sqrt(baseline_squared_sum)` のfloat。

## 使用場面

Convのfeature map全体を保存せず、pre-activation出力のrelative L2を集計するとき。

## 処理概要

1. 両方を有限な非負実数にする。
2. baselineの二乗和が0なら停止する。
3. 平方根の比を返す。batchごとの比は平均しない。

## 主なcontract / 注意事項

- 負値を0へ丸めない。分母0をepsilonへ置き換えない。
- boolとTensorは拒否する。
- サンプルごとのL2分布は返さない。それは `stage_change_summary` の対象で、こちらは二乗和だけを受け取る。

## 関連API

`relative_frobenius_error`, `stage_change_summary`
