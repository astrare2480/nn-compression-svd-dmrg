# `TTUpdateScope`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

1回の全データ更新で、TT pair と通常 parameter のどちらを更新するかを区別する。

## Signature

```python
TTUpdateScope(
    value: str,
) -> None
```

## 引数

- `value`: `"joint"`、`"tt_only"`、`"dense_only"` のいずれか。

## 戻り値

インスタンス。`value` に受け取った文字列を保持する。

## 使用場面

- 同じ初期状態と比較予算で、更新対象だけを変えるとき。
- 学習率 0 で固定を表さず、更新しない側を処理から外すとき。

## 処理概要

1. `value` が3つの文字列のどれかであることを確認する。
2. それ以外は `ValueError`。
3. クラス属性 `JOINT`、`TT_ONLY`、`DENSE_ONLY` は、それぞれの文字列。

## 主なcontract / 注意事項

- `joint` は結合 TT pair と通常 parameter の両方を更新する。
- `tt_only` は結合 TT pair だけを更新する。通常 parameter の勾配保存、更新候補、書き戻しはしない。forward は切断しない。
- `dense_only` は通常 parameter だけを更新する。TT の結合、SVD、rank 制限、core 書き戻し、中心移動、pair 巡回はしない。
- 文字列との等価比較ができる。

## 関連API

[[90_src設計/05_Core_API_v1/training/train_scoped_full_batch_budget]]、[[90_src設計/05_Core_API_v1/training/accumulate_scoped_full_batch_gradients]]、[[90_src設計/05_Core_API_v1/Internal_API]]
