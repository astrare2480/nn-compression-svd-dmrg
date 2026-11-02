# `tt_matrix_cores_to_tt_cores`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

4階TT-matrix coreの出力・入力mode $(m_k,n_k)$を1本の物理軸 $q_k=m_kn_k$ にまとめ、3階TT coreへ戻す。`tt_cores_to_tt_matrix_cores`の逆変換である。

## Signature

```python
tt_matrix_cores_to_tt_cores(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> list[torch.Tensor]
```

## 引数

- `cores`: 各要素がshape `(r_{k-1}, m_k, n_k, r_k)`のTT-matrix core列。
- `out_modes`: 各siteの出力mode $(m_1,\ldots,m_d)$。`cores[k].shape[1]`と一致させる。
- `in_modes`: 各siteの入力mode $(n_1,\ldots,n_d)$。`cores[k].shape[2]`と一致させる。

## 戻り値

各coreを `(r_{k-1}, m_k n_k, r_k)` へreshapeした新しいlist。値、dtype、device、autograd graphは維持する。

## 使用場面

- TT-matrixを、3階TT用の`tt_canonicalize`や`tt_canonicality_errors`へ渡すとき。
- site物理軸を `dense_to_tt_matrix_tensor` と同じ $a_k=i_kn_k+j_k$ の順序で1本に戻すとき。

## 処理概要

1. `validate_tt_matrix_cores`で4階coreとmodeの対応を確認する。
2. 各coreをC-orderで `(r_{k-1}, m_k n_k, r_k)` にreshapeする。
3. 新しいlistを返す。入力listと各Tensorは変更しない。

## 主なcontract / 注意事項

- SVDも正準化も行わない。bond次元は変えない。
- 左展開 `(r_{k-1} m_k n_k, r_k)` は、3階coreの左展開 `(r_{k-1} q_k, r_k)` と同一である。
- `q_k`の中では `n_k` が最も速く変わる。これは`tt_cores_to_tt_matrix_cores`が戻す順序と同じである。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/tt_cores_to_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/tt_mps/tt_canonicalize]]、[[90_src設計/05_Core_API_v1/tt_mps/canonicalize_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
