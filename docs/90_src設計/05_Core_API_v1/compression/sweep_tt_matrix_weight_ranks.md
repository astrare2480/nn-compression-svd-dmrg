# `sweep_tt_matrix_weight_ranks`

**Stability:** A
**定義:** `src/nn_compression/compression/tt_rank_sweep.py`

## 責務

dense weightを複数のrank上限でTT-matrix近似し、実rank、parameter数、再構成誤差をweight単位で記録する。model評価、DataFrame化、plot、候補選択は行わない。

## Signature
```python
sweep_tt_matrix_weight_ranks(W, out_modes, in_modes, ranks)
```

## 引数

- `W`: shape `(prod(out_modes), prod(in_modes))`のfloat32/float64 dense weight。
- `out_modes`: TT-matrixの出力mode列。
- `in_modes`: TT-matrixの入力mode列。
- `ranks`: boolでない正整数rank上限の非空列。重複と非昇順を許す。

## 戻り値

入力順の`list[dict]`。各recordは`requested_rank`, `actual_ranks`, `tt_parameters`, `dense_parameters`, `compression_ratio`, `absolute_error`, `relative_error`, `is_compressed`を持つ。

## 使用場面

Linear weightや行列化したConv weightについて、accuracy評価前にrank―誤差―parameter数の候補を作るとき。

## 処理概要

1. `W`を`max(abs(W))`で正規化する。
2. 各rankについて`dense_to_tt_matrix_cores`で分解する。
3. `tt_matrix_to_dense`で診断用に復元し誤差を測る。
4. core数と実rankをrecordへまとめる。

## 主なcontract / 注意事項

- `W.detach()`と`torch.no_grad()`を使い、autograd graphは作らない。
- 零行列とNaN/infを拒否する。入力は変更しない。
- `d=1`では内部bondがないため全rank候補が同じ結果になる。
- absolute errorがfloat64範囲を超える場合は`inf`。rankに対する誤差の単調性は保証しない。

## 関連API

`dense_to_tt_matrix_cores`, `tt_matrix_to_dense`, `sweep_tt_conv2d_weight_ranks`
