# `tt_matrix_shape_imbalance`

**Stability:** A
**定義:** `src/nn_compression/compression/tt_modes.py`

## 責務

TT-matrixのsite間shapeの偏りを、physical sizeの最大値と最小値の比として評価する。

## Signature
```python
tt_matrix_shape_imbalance(out_modes, in_modes)
```

## 引数

- `out_modes`: 出力側mode列。各siteの$m_k$を与える。
- `in_modes`: 入力側mode列。各siteの$n_k$を与える。

## 戻り値

$$
\frac{\max_k(m_kn_k)}{\min_k(m_kn_k)}
$$

をPython `float`で返す。`1.0`なら全siteのphysical sizeが等しい。

## 使用場面

同じdense shapeに対する複数のtensorization候補を、形状の均等さという一軸で比較するとき。

## 処理概要

`tt_matrix_physical_sizes`を呼び、得られた正整数列の`max/min`を計算する。

## 主なcontract / 注意事項

- 引数contractは`tt_matrix_physical_sizes`と同じ。
- 小さい値はaccuracy、圧縮率、最適rankを保証しない。このAPIは候補を選択しない。

## 関連API

`tt_matrix_physical_sizes`, `ordered_factorizations`, `sweep_tt_matrix_weight_ranks`
