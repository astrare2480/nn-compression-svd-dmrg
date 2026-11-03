# `initial_validation_selection`

**Stability:** A
**定義:** `src/nn_compression/training/validation_selection.py`

## 責務

最初の validation 観測を、strict best と stopping reference の両方の初期値にする。

## Signature

```python
initial_validation_selection(
    loss: float,
    step: int,
) -> ValidationSelection
```

## 引数

- `loss`: 最初の validation loss。
- `step`: その観測の step。0 以上の整数。

## 戻り値

非改善回数 0、early stopping 未成立の `ValidationSelection`。

## 使用場面

- epoch 0 または更新数 0 を best 候補に含めるとき。

## 処理概要

1. loss が有限であることを確認する。
2. その loss を strict best と reference の両方に置く。

## 主なcontract / 注意事項

- 非有限値は `ValueError`。停止理由 `non_finite` へは変換しない。
- 初期化だけでは patience を消費しない。

## 関連API

- `ValidationSelection`
- `update_validation_selection`
