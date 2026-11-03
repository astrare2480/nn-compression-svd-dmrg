# `update_validation_selection`

**Stability:** A
**定義:** `src/nn_compression/training/validation_selection.py`

## 責務

新しい validation loss を、strict best と stopping reference へ別々に反映する。

## Signature

```python
update_validation_selection(
    state: ValidationSelection,
    loss: float,
    step: int,
    *,
    min_delta: float,
    patience: int,
) -> ValidationSelection
```

## 引数

- `state`: 直前の選択状態。
- `loss`: 新しい validation loss。
- `step`: その観測の step。
- `min_delta`: reference を動かすために必要な低下量。
- `patience`: 非改善がこの回数に達したら early stopping。

## 戻り値

更新後の `ValidationSelection`。

## 使用場面

- 小さな改善でも best checkpoint は更新し、停止判定は `min_delta` を使うとき。

## 処理概要

1. loss が strict best より低ければ、best の loss と step を更新する。
2. loss が以前の reference より厳密に低く、低下量が `min_delta` 以上なら、reference を現在の loss にし、非改善回数を 0 にする。同値は `min_delta=0` でも改善にしない。
3. それ以外は非改善回数を 1 増やす。回数が `patience` 以上なら early stopping を成立させる。

## 主なcontract / 注意事項

- 低下が `min_delta` 未満でも strict best は更新する。reference は動かさない。
- reference は、小さな改善のたびに追従しない。前回の reference からの累積低下で判定する。
- 同値の loss は strict best を更新しない。
- 非有限 loss は `ValueError`。成立後の追加観測は `RuntimeError`。

## 関連API

- `initial_validation_selection`
- `replay_validation_selection`
