# `tt_cores_to_tt_matrix_cores`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_matrix.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/15_ttlinear_dense_forward_equivalence_learning.ipynb`

## 責務

TT-SVDが返す3階TT coreのsite物理軸$q_k$を$(m_k,n_k)$へ分け、4階TT-matrix coreへ変換する。

## Signature

```python
tt_cores_to_tt_matrix_cores(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> list[torch.Tensor]
```

## 引数

- `cores`: 各要素がshape `(r_{k-1}, q_k, r_k)`の通常TT core列。`q_k = out_modes[k] * in_modes[k]`を満たす必要がある。
- `out_modes`: 各siteの出力側物理次元$(m_1,\ldots,m_d)$。
- `in_modes`: 各siteの入力側物理次元$(n_1,\ldots,n_d)$。`cores`および`out_modes`と同じ長さにする。

## 戻り値

各要素が

$$
(r_{k-1},q_k,r_k)
\longrightarrow
(r_{k-1},m_k,n_k,r_k)
$$

へreshapeされた新しいlistを返す。Tensor値、dtype、device、autograd graphは維持する。

## 使用場面

- `tt_svd_exact`または`tt_svd`の結果をTT-matrix APIへ接続するとき。
- site単位のcombined物理軸を出力・入力axisへ戻すとき。

## 処理概要

1. `validate_tt_cores`で通常TT core列を検証する。
2. 対応dtypeとmode列を検証する。
3. 各siteで$q_k=m_kn_k$を確認する。
4. 各coreを4階へreshapeし、新しいlistへ格納する。

## 主なcontract / 注意事項

- TT-SVDは実行せず、rankも変更しない。
- 入力listと各Tensorをin-place変更しない。
- `cores`のsite数と両mode列の長さを一致させる。
- 物理軸$q_k$の要素順は`dense_to_tt_matrix_tensor`が作る$a_k=i_kn_k+j_k$の順序を前提とする。

## 関連API

[[90_src設計/05_Core_API_v1/compression/dense_to_tt_matrix_tensor]]、[[90_src設計/05_Core_API_v1/compression/dense_to_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/compression/tt_svd_exact]]、[[90_src設計/05_Core_API_v1/Internal_API]]
