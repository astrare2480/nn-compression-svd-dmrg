# `mlp_logits_error_bound`

**Stability:** A
**定義:** `src/nn_compression/metrics/error_bounds.py`

## 責務

`Linear -> ReLU -> Linear`で第1層weightだけが変化したとき、batch全体のlogits Frobenius誤差上界を返す。

## Signature
```python
mlp_logits_error_bound(X, delta_W1, W2)
```

## 引数

- `X`: shape `(batch, in_features)`の入力。
- `delta_W1`: shape `(hidden_features, in_features)`の第1層weight差。
- `W2`: shape `(num_classes, hidden_features)`の変更しない第2層weight。

## 戻り値

$$
\lVert X\rVert_F\lVert\Delta W_1\rVert_2\lVert W_2\rVert_2
$$

を表す0階Tensor。

## 使用場面

第1 LinearをTT化したMLPで、圧縮後logits誤差の理論上界を評価するとき。

## 処理概要

ReLUの1-Lipschitz性と行列normの劣乗法性を使い、入力norm、摂動のspectral norm、第2層のspectral normを掛ける。

## 主なcontract / 注意事項

- 3因子のいずれかが厳密な零行列なら結果も厳密な0。
- float32/float64、有限、同一dtype/device、整合するfeature数を要求する。
- 非零入力で表現範囲を超えた上界は`inf`。有限入力からNaNを返さない。

## 関連API

`mlp_sample_logits_error_bounds`, `batch_stability_certificate`
