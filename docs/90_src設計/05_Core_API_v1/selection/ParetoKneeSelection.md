# `ParetoKneeSelection`

**Stability:** A
**定義:** `src/nn_compression/selection/pareto_knee.py`

## 責務

Pareto frontierとknee選択の成否を、未定義理由を失わず保持する不変dataclass。

## Signature
```python
ParetoKneeSelection(frontier, frontier_source_index, is_defined, knee_index, knee_row, knee_distance, undefined_reason)
```

## 引数

- `frontier`: x昇順へ整列したPareto frontierのDataFrame。
- `frontier_source_index`: frontier各行に対応する入力DataFrameのindex label列。
- `is_defined`: kneeを一意かつ安全に定義できたか。
- `knee_index`: 入力DataFrame上のknee index。未定義なら`None`。
- `knee_row`: kneeの行copy。未定義なら`None`。
- `knee_distance`: 正規化座標上の直線距離。未定義なら`None`。
- `undefined_reason`: 未定義理由。定義時は`None`。

## 戻り値

7 fieldを持つfrozen instance。DataFrameをfieldに持つためdataclass自体の`eq`は無効。

## 使用場面

rank sweepの候補表から、kneeが存在する場合と判断不能な場合を同じ型で扱うとき。

## 処理概要

値の生成とvalidationは`select_pareto_knee`が担当する。

## 主なcontract / 注意事項

- 未定義時も`frontier`は常に返す。
- 理由は`too_few_points`, `zero_x_width`, `zero_y_width`, `nearly_collinear`, `undesirable_side`, `ambiguous_tie`のいずれか。

## 関連API

`select_pareto_knee`, `extract_pareto_frontier`
