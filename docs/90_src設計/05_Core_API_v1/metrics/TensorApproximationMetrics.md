# `TensorApproximationMetrics`

**Stability:** A
**定義:** `src/nn_compression/metrics/tensor_comparison.py`

## 責務

Tensor近似の絶対Frobenius誤差、相対誤差、RMSE、最大絶対誤差を保持する不変dataclass。

## Signature
```python
TensorApproximationMetrics(absolute_frobenius_error, relative_frobenius_error, rmse, max_abs_error)
```

## 引数

- `absolute_frobenius_error`: $\lVert A-\hat A\rVert_F$。
- `relative_frobenius_error`: $\lVert A-\hat A\rVert_F/\lVert A\rVert_F$。
- `rmse`: 全要素差の二乗平均平方根。
- `max_abs_error`: 要素ごとの差の最大絶対値。

## 戻り値

4つのPython `float` fieldを持つfrozen instance。

## 使用場面

weight、activation、kernelなどの近似品質を同一schemaで記録するとき。

## 処理概要

値の計算は`tensor_approximation_metrics`が担当する。

## 主なcontract / 注意事項

- `dataclasses.asdict`でDataFrameの1行に変換できる。
- 数学的な値がfloat64範囲外ならfieldは`inf`になり得る。

## 関連API

`tensor_approximation_metrics`
