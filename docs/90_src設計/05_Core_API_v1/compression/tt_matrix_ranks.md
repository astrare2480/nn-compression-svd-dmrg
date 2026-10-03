# `tt_matrix_ranks`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_matrix.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb`

## 責務

4階TT-matrix core列から、左右の境界rankを含むTT-rank列を取得する。

## Signature

```python
tt_matrix_ranks(
    cores: Sequence[torch.Tensor],
) -> tuple[int, ...]
```

## 引数

- `cores`: 各要素がshape `(r_{k-1}, m_k, n_k, r_k)`のTT-matrix core列。境界rank、隣接bond、dtype、deviceを含む構造contractを満たす必要がある。

## 戻り値

長さ$d+1$のtuple

$$
(r_0,r_1,\ldots,r_d)
$$

を返す。境界条件により$r_0=r_d=1$である。形式は通常の3階TTに対する`tt_ranks`とそろえる。

## 使用場面

- TT-matrixの内部bond dimensionを記録するとき。
- `max_rank`による打ち切り結果を確認するとき。
- dense行列とのparameter数・圧縮率比較にrank情報を添えるとき。

## 処理概要

1. `validate_tt_matrix_cores`でcore列を検証する。
2. 各coreの左bond `shape[0]`を順に取得する。
3. 最終coreの右bond `shape[-1]`を末尾へ追加してtupleで返す。

## 主なcontract / 注意事項

- 内部rankだけでなく、両端のrank 1も含める。
- coreを変更せず、Tensor値の計算も行わない。
- 空列、4階でないcore、bond不整合、dtype/device不整合は受理しない。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_matrix_num_parameters]]、[[90_src設計/05_Core_API_v1/compression/tt_ranks]]、[[90_src設計/05_Core_API_v1/Internal_API]]
