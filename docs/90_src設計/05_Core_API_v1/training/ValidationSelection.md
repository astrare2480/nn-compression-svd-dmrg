# `ValidationSelection`

**Stability:** A
**定義:** `src/nn_compression/training/validation_selection.py`

## 責務

1 回の validation 観測後の、strict best と stopping reference の状態を保持する。

## Signature

```python
ValidationSelection(
    strict_best_loss: float,
    strict_best_step: int,
    reference_loss: float,
    non_improvement_count: int,
    early_stop_triggered: bool,
) -> None
```

## 引数

- `strict_best_loss`: これまでの最小 validation loss。
- `strict_best_step`: その loss を記録した step。
- `reference_loss`: patience の基準になる loss。
- `non_improvement_count`: reference から `min_delta` 以上下がっていない連続回数。
- `early_stop_triggered`: その回数が patience に達したか。

## 戻り値

インスタンスを返す。計算はしない。

## 使用場面

- best checkpoint の位置と、early stopping の基準を別々に持つとき。

## 処理概要

1. 渡された値を float または int または bool として保持する。

## 主なcontract / 注意事項

- このオブジェクトを作っただけでは停止判定は行われない。
- 同値の loss は、より早い step の strict best を維持する側で使う。

## 関連API

- `initial_validation_selection`
- `update_validation_selection`
- `replay_validation_selection`
