# `incomplete_scope_names`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

3条件の比較図を書いてよいかを判定し、未完了の scope 名を返す。

## Signature

```python
incomplete_scope_names(
    observations: Mapping,
    scopes: Sequence[str],
    total_updates: int,
) -> list[str]
```

## 引数

- `observations`: scope 名から観測への対応。集計行が無い scope は `None`。
- `scopes`: 比較に必要な scope の順。
- `total_updates`: 正常完了に必要な更新数。

## 戻り値

未完了 scope の名前。空リストのときだけ比較図と表を作ってよい。

## 使用場面

- 1条件または2条件だけ完了している途中状態で、比較セルを SKIP するとき。
- 3条件がそろったあとだけ図を書くとき。

## 処理概要

1. 観測が `None` の scope は未完了として名前を集める。
2. 観測がある場合は、設定一致、history の存在、`max_updates`、成功更新数、最終更新数を確認する。
3. 観測があるのに内容が不正なら `ValueError`。SKIP にはしない。

## 主なcontract / 注意事項

- この関数はファイルを読まない。削除も上書きもしない。
- 未完了は戻り値、不正な記録は例外で区別する。
- 観測にある `settings_match` と `history_exists` は真偽値。`True` 以外は不正または未完了の判定に使う。

## 関連API

[[90_src設計/05_Core_API_v1/training/assert_scope_budget_completed]]、[[90_src設計/05_Core_API_v1/Internal_API]]
