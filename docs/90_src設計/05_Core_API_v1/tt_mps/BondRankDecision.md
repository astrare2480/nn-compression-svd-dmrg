# `BondRankDecision`

**Stability:** A
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/08_adaptive_bond_rank_dmrg_like_pareto.ipynb`

## 責務

相対 discarded weight による bond rank 選択の記録を保持する。数値的 rank と、SVD 行列の shape から決まる保持可能本数を分けて残す。

## Signature

```python
BondRankDecision(
    available_rank: int,
    threshold_required_rank: int,
    retained_rank: int,
    relative_discarded_weight: float,
    threshold_satisfied: bool,
    cap_limited: bool,
    min_rank_limited: bool,
    zero_spectrum: bool,
) -> None
```

## 引数

- `available_rank`: 還元 SVD の本数。行列 shape の `min(行, 列)` であり、数値的 rank ではない。
- `threshold_required_rank`: 上限を掛ける前に、閾値を満たす最小 rank。探索範囲は 1 から `available_rank`。
- `retained_rank`: 下限、上限、形状制約を適用したあとの保持 rank。
- `relative_discarded_weight`: `retained_rank` で捨てた特異値二乗和を、全特異値二乗和で割った値。
- `threshold_satisfied`: 実際の切り捨て量が閾値以下か。上限で届かない場合は偽。
- `cap_limited`: `max_bond_rank` が必要 rank を制限したか。
- `min_rank_limited`: `min_bond_rank` が必要 rank より大きくしたか。
- `zero_spectrum`: 有限な特異値がすべて 0 か。NaN や Inf はこの値にしない。

## 戻り値

上記フィールドを持つ frozen dataclass。

## 使用場面

- 2-core 更新後の SVD で、固定 rank ではなく閾値から bond 次元を残すとき。

## 処理概要

1. 選択関数が特異値列から各フラグを計算する。
2. 呼び出し側は `retained_rank` で core を作り、残りの項目をログへ残す。

## 主なcontract / 注意事項

- `relative_discarded_weight` は相対 Frobenius 誤差そのものではなく、その二乗の割合である。
- 上限のため閾値を満たせないとき、`threshold_satisfied` を真にしない。
- ゼロスペクトルは `threshold_required_rank=1`、`retained_rank=min_bond_rank`、切り捨て量 0、`threshold_satisfied=True`。`min_bond_rank > 1` のとき `min_rank_limited` を真にする。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/select_bond_rank_from_singular_values]]、[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix_by_discarded_weight]]、[[90_src設計/05_Core_API_v1/Internal_API]]
