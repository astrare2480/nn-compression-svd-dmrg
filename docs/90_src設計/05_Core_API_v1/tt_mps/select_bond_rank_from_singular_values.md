# `select_bond_rank_from_singular_values`

**Stability:** A
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/08_adaptive_bond_rank_dmrg_like_pareto.ipynb`

## 責務

降順の特異値から、相対 discarded weight が閾値以下になる最小 bond rank を選ぶ。

## Signature

```python
select_bond_rank_from_singular_values(
    singular_values: torch.Tensor,
    *,
    threshold: float,
    min_bond_rank: int,
    max_bond_rank: int,
    comparison_atol: float = 0.0,
    comparison_rtol: float = 0.0,
) -> BondRankDecision
```

## 引数

- `singular_values`: 降順で 0 以上の 1 階特異値。長さが `available_rank`。
- `threshold`: 相対 discarded weight の上限。0 以上の有限値。
- `min_bond_rank`: 保持 rank の下限。1 以上。
- `max_bond_rank`: 保持 rank の上限。`min_bond_rank` 以上。
- `comparison_atol`: 比較に足す絶対許容。既定 0。
- `comparison_rtol`: 閾値の絶対値に掛ける相対許容。既定 0。

## 戻り値

`BondRankDecision`。

## 使用場面

- 既知の特異値列で rank 規則だけを確認するとき。
- 結合テンソルの SVD 直後に、固定 `max_bond` とは別の規則で rank を決めるとき。

## 処理概要

1. NaN、Inf、負の特異値は拒否する。
2. すべて 0 なら `threshold_required_rank=1`、`retained_rank=min_bond_rank`、切り捨て量 0、ゼロスペクトルとする。`min_bond_rank > 1` のとき `min_rank_limited` を真にする。
3. 最大特異値でスケールしてから float64 の二乗を取り、小さい側からの逆向き累積和で tail を作る。閾値以下の最小 r を探す。
4. 下限と上限を適用し、実際の切り捨て量で `threshold_satisfied` を決める。

## 主なcontract / 注意事項

- 比較は `discarded_weight <= threshold + comparison_atol + comparison_rtol * abs(threshold)`。既定の許容は 0 であり、小さい閾値を大きな絶対許容で吸収しない。
- 二乗和は最大特異値でスケールし、tail は rank より後ろの二乗和になる。途中と最終の相対 discarded weight が非有限ならエラー。
- この計算は rank 選択専用の dtype 変換であり、モデル parameter の dtype は変えない。
- `DELTA=0` は、厳密に 0 の特異値を捨てる最小 rank であり、指定した固定 rank を常に保つ処理ではない。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/BondRankDecision]]、[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix_by_discarded_weight]]、[[90_src設計/05_Core_API_v1/Internal_API]]
