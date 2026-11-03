# `replay_validation_selection`

**Stability:** A
**定義:** `src/nn_compression/training/validation_selection.py`

## 責務

保存済みの validation loss 列を先頭から再生し、各時点の選択状態を返す。

## Signature

```python
replay_validation_selection(
    losses: Sequence[float],
    steps: Sequence[int],
    *,
    min_delta: float,
    patience: int,
) -> list
```

## 引数

- `losses`: 時系列の validation loss。
- `steps`: 各 loss に対応する step。
- `min_delta`: reference を動かす低下量。
- `patience`: early stopping の連続非改善回数。

## 戻り値

各観測後の `ValidationSelection`。early stopping が成立した時点で打ち切る。

## 使用場面

- 過去の履歴に early stopping が無かったことを、同じ規則で確認するとき。
- 継続学習の直前に、非改善回数と reference を復元するとき。

## 処理概要

1. 先頭を `initial_validation_selection` で初期化する。
2. 残りを順に `update_validation_selection` へ渡す。
3. early stopping が成立したら、その状態までを返す。

## 主なcontract / 注意事項

- 履歴へ新しい行は足さない。
- 非有限値があれば、その位置で `ValueError` になる。

## 関連API

- `update_validation_selection`
