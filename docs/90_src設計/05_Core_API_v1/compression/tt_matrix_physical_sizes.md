# `tt_matrix_physical_sizes`

**Stability:** A
**定義:** `src/nn_compression/compression/tt_modes.py`

## 責務

TT-matrixの各siteで、出力modeと入力modeを組にしたphysical size $q_k=m_kn_k$ を返す。候補を選択せず、tensorizationの形状評価に必要な量だけを計算する。

## Signature
```python
tt_matrix_physical_sizes(out_modes, in_modes)
```

## 引数

- `out_modes`: 出力側mode $(m_1,\ldots,m_d)$。空でない正整数列。
- `in_modes`: 入力側mode $(n_1,\ldots,n_d)$。`out_modes`と同じsite数の正整数列。

## 戻り値

`(m_1*n_1, ..., m_d*n_d)`を表す`tuple[int, ...]`。

## 使用場面

tensorization候補のsite形状を比較するとき、またはTT-SVDへ渡すcombined physical axisを確認するとき。

## 処理概要

`validate_tt_matrix_modes`で両mode列を正規化・検証し、siteごとの積を順に作る。例えば`out_modes=(2, 3)`、`in_modes=(4, 5)`なら`(8, 15)`を返す。

## 主なcontract / 注意事項

- bool、非整数、非正値、空列、site数不一致を拒否する。
- modeの順序はTTのbond cutを決めるため、並べ替えない。
- 推奨tensorizationの自動選択は責務外である。

## 関連API

`tt_matrix_shape_imbalance`, `tt_matrix_num_parameters_from_ranks`, `dense_to_tt_matrix_tensor`
