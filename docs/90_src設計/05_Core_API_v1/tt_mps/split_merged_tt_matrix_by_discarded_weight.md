# `split_merged_tt_matrix_by_discarded_weight`

**Stability:** A
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/08_adaptive_bond_rank_dmrg_like_pareto.ipynb`

## 責務

結合テンソルを 1 回だけ還元 SVD し、相対 discarded weight で選んだ rank と方向で 2 core に戻す。

## Signature

```python
split_merged_tt_matrix_by_discarded_weight(
    merged: torch.Tensor,
    *,
    threshold: float,
    min_bond_rank: int,
    max_bond_rank: int,
    direction: str = "right",
    comparison_atol: float = 0.0,
    comparison_rtol: float = 0.0,
) -> tuple[TTMatrixPairSplit, BondRankDecision]
```

## 引数

- `merged`: shape `(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})` の結合テンソル。
- `threshold`: 相対 discarded weight の上限。
- `min_bond_rank`: 保持 rank の下限。
- `max_bond_rank`: 保持 rank の上限。
- `direction`: `"right"` または `"left"`。特異値を置く側と、直交中心の移動先は `split_merged_tt_matrix` と同じ。
- `comparison_atol`: 比較の絶対許容。既定 0。
- `comparison_rtol`: 比較の相対許容。既定 0。

## 戻り値

`TTMatrixPairSplit` と `BondRankDecision`。分割の `retained_rank` は決定の `retained_rank` と一致する。

## 使用場面

- 固定 `max_bond` ではなく、更新後の結合テンソルごとに bond rank を選び直すとき。

## 処理概要

1. `merged` を左右 3 軸で行列化し、`full_matrices=False` の SVD を 1 回行う。
2. `select_bond_rank_from_singular_values` で rank を決める。
3. 選んだ本数だけを、指定方向の core へ戻す。

## 主なcontract / 注意事項

- 固定 rank の既存経路は `split_merged_tt_matrix` である。この関数はそれを置き換えない。
- 過去に縮めた bond を次回の上限にはしない。次回は、その時点の結合テンソルの SVD から選び直す。
- 外側 bond と physical mode による shape 制約は、SVD の本数として既に入っている。
- 入力は変更しない。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/select_bond_rank_from_singular_values]]、[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix]]、[[90_src設計/05_Core_API_v1/tt_mps/BondRankDecision]]、[[90_src設計/05_Core_API_v1/Internal_API]]
