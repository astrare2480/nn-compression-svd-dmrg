# `load_condition_summary`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/02_full_batch_sweep_update_budget_grid.ipynb`

## 責務

条件グリッドの集計CSVを読み、`condition_id`の重複を拒否する。

## Signature

```python
load_condition_summary(
    path: Path | str,
) -> list[dict[str, str]]
```

## 引数

- `path`: 集計CSV。ファイルが無ければ空のリストを返す。

## 戻り値

CSVの各行。値は文字列である。

## 使用場面

- 12条件の再実行前に、どこまで記録済みかを読むとき。
- 実行スイッチがオフのとき、保存済みの集計だけを表示するとき。

## 処理概要

1. ファイルが無ければ空リストを返す。
2. `condition_id`が空、または同じIDが2行以上あれば`ValueError`にする。

## 主なcontract / 注意事項

- この関数はファイルを変更しない。
- 重複したIDを黙って片方だけ採用しない。

## 関連API

[[90_src設計/05_Core_API_v1/training/run_resumable_condition_grid]]、[[90_src設計/05_Core_API_v1/Internal_API]]
