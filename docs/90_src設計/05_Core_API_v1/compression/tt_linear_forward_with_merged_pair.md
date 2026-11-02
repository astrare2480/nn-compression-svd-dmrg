# `tt_linear_forward_with_merged_pair`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_linear_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

隣接pairの2 coreの値の代わりに6階の結合テンソルを使い、dense weightを作らずに $Y=XW^{\mathsf T}+b$ を計算する。

## Signature

```python
tt_linear_forward_with_merged_pair(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    pair_index: int,
    merged: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    bias: torch.Tensor | None = None,
) -> torch.Tensor
```

## 引数

- `x`: shape `(batch, in_features)`の入力。`tt_linear_forward`と同じ契約。
- `cores`: 層のTT-matrix core列。pair以外のsiteの値を縮約に使う。
- `pair_index`: 結合した左coreのsite index。`0 <= pair_index <= d-2`。
- `merged`: shape `(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})`の結合テンソル。
- `out_modes`: 出力mode $(m_1,\ldots,m_d)$。
- `in_modes`: 入力mode $(n_1,\ldots,n_d)$。
- `bias`: `None`またはshape `(out_features,)`のbias。

## 戻り値

shape `(batch, out_features)`の出力。`x`、`merged`、pair以外のcore、`bias`のautograd graphをつなぐ。

## 使用場面

- 結合テンソルを学習変数にして、その層の線形写像を計算するとき。
- 元の2 coreによる`tt_linear_forward`と、結合テンソル経由の出力が一致するかを確認するとき。

## 処理概要

1. core列、入力、bias、`merged`のshapeを確認する。
2. `tt_linear_forward`と同じ軸順で、pairより前のsiteを1 coreずつ縮約する。
3. pairでは $T$ の軸 $(\alpha,i_k,j_k,i_{k+1},j_{k+1},\gamma)$ のうち $\alpha,j_k,j_{k+1}$ を一度に縮約する。
4. pairより後のsiteを通常どおり縮約し、右端bondを落としてbiasを加える。

## 主なcontract / 注意事項

- `cores[pair_index]`と`cores[pair_index+1]`の値は使わない。左右のbondとmodeが`merged`と合うかの検証にだけ使う。
- denseな $W$ は作らない。
- `detach`も`torch.no_grad`も使わない。
- `merged`がちょうど2 coreの縮約なら、戻り値は`tt_linear_forward`と丸め誤差の範囲で一致する。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_linear_forward]]、[[90_src設計/05_Core_API_v1/tt_mps/merge_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/compression/module_forward_with_merged_tt_linear_pair]]、[[90_src設計/05_Core_API_v1/Internal_API]]
