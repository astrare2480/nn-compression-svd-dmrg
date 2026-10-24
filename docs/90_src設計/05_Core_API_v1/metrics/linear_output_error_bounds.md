# `linear_output_error_bounds`

**Stability:** A
**定義:** `src/nn_compression/metrics/error_bounds.py`

## 責務

$\Delta Y=X\Delta W^{\mathsf T}$について、$\lVert\Delta Y\rVert_F$の2つの安全な上界を計算する。

## Signature
```python
linear_output_error_bounds(X, delta_W)
```

## 引数

- `X`: shape `(batch, in_features)`の入力行列。
- `delta_W`: shape `(out_features, in_features)`の重み差`W_approx-W`。

## 戻り値

`LinearOutputErrorBounds`。各fieldは入力と同じdtype/deviceの0階Tensor。

## 使用場面

TT、SVD、量子化などによるweight近似誤差を、実際の出力誤差と比較するとき。

## 処理概要

$$
\lVert\Delta Y\rVert_F\leq\lVert X\rVert_F\lVert\Delta W\rVert_2,
\qquad
\lVert\Delta Y\rVert_F\leq\lVert X\rVert_2\lVert\Delta W\rVert_F
$$

をそれぞれ計算する。どちらかが厳密な零行列なら、相手のnormを評価する前に入力へ接続された0を返す。

## 主なcontract / 注意事項

- float32/float64、有限、非空、同一dtype/deviceを要求する。
- 有限入力からNaNを返さない。非零因子のnormがoverflowした場合は`inf`。
- 通常範囲のautogradを維持し、入力を変更しない。

## 関連API

`LinearOutputErrorBounds`, `mlp_logits_error_bound`, `mlp_sample_logits_error_bounds`
