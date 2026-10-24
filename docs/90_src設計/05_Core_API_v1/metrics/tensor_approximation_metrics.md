# `tensor_approximation_metrics`

**Stability:** A
**定義:** `src/nn_compression/metrics/tensor_comparison.py`

## 責務

同shape Tensorの近似誤差を、DataFrameへ保存可能な4指標へまとめる。

## Signature
```python
tensor_approximation_metrics(reference, approximation)
```

## 引数

- `reference`: 基準となる任意階数の有限float32/float64 Tensor。
- `approximation`: 基準とshape/dtype/deviceが一致する近似Tensor。

## 戻り値

`TensorApproximationMetrics`。

## 使用場面

TT/SVD/Tuckerによるweightや中間出力の誤差を共通schemaで比較するとき。

## 処理概要

絶対誤差系は両Tensorの共通な2のべき乗scaleで計算する。相対誤差は通常経路では同じscaleを使い、極端な桁差で非零referenceがscale後にunderflowする場合だけ、分子・分母を独立した仮数部と指数部で評価する。

## 主なcontract / 注意事項

- 厳密な零referenceでは相対誤差未定義のため`ValueError`。
- 非零referenceを内部underflowで零と誤認しない。真の比がfloat64範囲を超える場合は`inf`で、有限入力からNaNを作らない。
- 空、NaN/inf、shape/dtype/device不一致を拒否し、autogradを契約に含めない。

## 関連API

`TensorApproximationMetrics`, `relative_frobenius_error`, `compare_logits`
