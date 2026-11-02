# `merge_tt_matrix_cores`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

隣接する2つの4階TT-matrix coreを、共有bondで縮約して6階の結合テンソルにする。物理DMRGの3階MPS結合とは軸数が違うため、`merge_two_sites`は使わない。

## Signature

```python
merge_tt_matrix_cores(
    left_core: torch.Tensor,
    right_core: torch.Tensor,
) -> torch.Tensor
```

## 引数

- `left_core`: $G_k$。shape `(r_{k-1}, m_k, n_k, r_k)`。
- `right_core`: $G_{k+1}$。shape `(r_k, m_{k+1}, n_{k+1}, r_{k+1})`。左bondは`left_core`の右bondと一致させる。

## 戻り値

結合テンソル $T$。shape `(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})`。

$$
T[\alpha,i_k,j_k,i_{k+1},j_{k+1},\gamma]
=
\sum_{\beta}
G_k[\alpha,i_k,j_k,\beta]\,
G_{k+1}[\beta,i_{k+1},j_{k+1},\gamma]
$$

dtype、device、autograd graphは入力を維持する。

## 使用場面

- 隣接2 coreを、分類損失で更新する1個の学習変数にする前の縮約。
- 再分割した2 coreが元の結合テンソルを再現するかの確認。

## 処理概要

1. 両coreが4階であること、dtype、device、共有bondを確認する。
2. 右端軸と左端軸を`torch.tensordot`で縮約する。
3. 入力Tensorは変更しない。`detach`しない。

## 主なcontract / 注意事項

- 独立な学習変数が欲しいときは、呼び出し側が戻り値を明示して`detach().clone()`する。この関数の中では切らない。
- 物理DMRGの`merge_two_sites`は3階core `(r, n, r)`を4階 `(r, n, n, r)`にする。ここでの戻り値は6階である。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/merged_tt_matrix_to_matrix]]、[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix]]、[[90_src設計/05_Core_API_v1/compression/tt_linear_forward_with_merged_pair]]、[[90_src設計/05_Core_API_v1/Internal_API]]
