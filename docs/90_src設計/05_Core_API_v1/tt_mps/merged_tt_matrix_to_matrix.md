# `merged_tt_matrix_to_matrix`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

6階の結合テンソルを、左3軸と右3軸でまとめた行列にする。SVDの入力と、切り捨て残差の計算に使う。

## Signature

```python
merged_tt_matrix_to_matrix(
    merged: torch.Tensor,
) -> torch.Tensor
```

## 引数

- `merged`: shape `(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})`の結合テンソル。

## 戻り値

行列 $M$。shape `(r_{k-1} m_k n_k,\ m_{k+1} n_{k+1} r_{k+1})`。

$$
M[(\alpha,i_k,j_k),\ (i_{k+1},j_{k+1},\gamma)]
=
T[\alpha,i_k,j_k,i_{k+1},j_{k+1},\gamma]
$$

`reshape`のみなのでautograd graphは維持する。

## 使用場面

- 結合テンソルの還元SVDの直前。
- 再結合したテンソルとのFrobenius残差を、同じ行列形で比べるとき。

## 処理概要

1. 6階であること、各次元が正であること、dtypeがfloat32またはfloat64であることを確認する。
2. 先頭3軸と末尾3軸をそれぞれ1次元にまとめてreshapeする。

## 主なcontract / 注意事項

- 入力は変更しない。`detach`しない。
- この行列化は軸の並べ替えを含まない。$T$の軸順が`merge_tt_matrix_cores`と同じであることが前提である。
- denseな重み $W$ への写像ではない。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/merge_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix]]、[[90_src設計/05_Core_API_v1/Internal_API]]
