# `reject_incomplete_scope_rows`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

集計に残っている対象 scope が正常完了でないとき、再開によるスキップの前に止める。

## Signature

```python
reject_incomplete_scope_rows(
    rows: Sequence[Mapping],
    scopes: Sequence[str],
    total_updates: int,
) -> None
```

## 引数

- `rows`: 既存の集計行。ファイルが無ければ空でよい。
- `scopes`: 今回の条件名。これ以外の行は見ない。
- `total_updates`: 要求した更新予算。

## 戻り値

なし。対象行がすべて正常完了のときだけ返る。

## 使用場面

- 再開関数を呼ぶ前に、異常停止の行を完了済みとしてスキップしたくないとき。

## 処理概要

1. `scopes` に含まれる `condition_id` だけを見る。
2. 各行の停止理由と成功更新数を `assert_scope_budget_completed` へ渡す。
3. 行が無い条件は、ここでは未実行として扱う。

## 主なcontract / 注意事項

- この関数はディレクトリを削除も上書きもしない。
- 文字列で読んだ更新数は数値へ直してから判定する。
- 対象外の `condition_id` は判定しない。

## 関連API

[[90_src設計/05_Core_API_v1/training/assert_scope_budget_completed]]、[[90_src設計/05_Core_API_v1/training/load_condition_summary]]、[[90_src設計/05_Core_API_v1/Internal_API]]
