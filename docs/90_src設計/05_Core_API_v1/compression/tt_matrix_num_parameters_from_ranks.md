# `tt_matrix_num_parameters_from_ranks`

**Stability:** A
**定義:** `src/nn_compression/compression/tt_modes.py`

## 責務

coreを生成せず、mode列とTT-rank列からTT-matrix coreの総parameter数を計算する。

## Signature
```python
tt_matrix_num_parameters_from_ranks(out_modes, in_modes, ranks)
```

## 引数

- `out_modes`: 出力側mode $(m_1,\ldots,m_d)$。
- `in_modes`: 入力側mode $(n_1,\ldots,n_d)$。
- `ranks`: 境界を含むrank列 $(r_0,\ldots,r_d)$。長さは$d+1$、全要素は正整数、$r_0=r_d=1$。

## 戻り値

$$
\sum_{k=1}^{d} r_{k-1}m_kn_kr_k
$$

をPython `int`で返す。

## 使用場面

rank候補を分解前に絞り込むとき、または生成済みcoreのparameter数と理論値を照合するとき。

## 処理概要

mode列とrank列を検証し、siteごとのcore要素数を合計する。

## 主なcontract / 注意事項

- NumPy整数は受理するが、boolとfloatは整数として扱わない。
- rank長不一致、非正rank、境界rank不一致は`ValueError`。
- 実在coreに対する`tt_matrix_num_parameters`と同じ値になる。

## 関連API

`tt_matrix_num_parameters`, `tt_matrix_physical_sizes`, `sweep_tt_matrix_weight_ranks`
