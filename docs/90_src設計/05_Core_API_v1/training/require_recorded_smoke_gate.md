# `require_recorded_smoke_gate`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

保存済み smoke が現在の識別子と合格条件に一致しないとき、主実験を始めずに止める。

## Signature

```python
require_recorded_smoke_gate(
    report: Mapping,
    expected: Mapping,
    fields: Sequence[str],
) -> None
```

## 引数

- `report`: 保存済み smoke の記録。
- `expected`: 現在の checkpoint、データ、コード、設定、合格フラグ。
- `fields`: 一致を要求するキー。

## 戻り値

なし。一致したときだけ返る。

## 使用場面

- smoke のあと、同じ checkpoint とコードで主実験を始める前。
- 記録が無い、または合格していない実行を主実験へ進めたくないとき。

## 処理概要

1. `fields` のどれかが記録に無ければ、未完了として `RuntimeError`。
2. 記録と期待が違えば、項目を示した `RuntimeError`。
3. ファイルの削除や上書きはしない。

## 主なcontract / 注意事項

- この関数は学習を始めない。呼び出し側が、返ったあとだけ主実験を呼ぶ。
- 合格フラグが偽の記録は、期待が真である限り不一致になる。
- 閾値を緩める処理は含まない。

## 関連API

[[90_src設計/05_Core_API_v1/training/train_scoped_full_batch_budget]]、[[90_src設計/05_Core_API_v1/Internal_API]]
