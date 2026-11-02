# `assert_scope_budget_completed`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

固定更新予算の条件が正常完了したときだけ返し、それ以外は集計行を追加する前に止める。

## Signature

```python
assert_scope_budget_completed(
    stop_reason: str,
    completed_updates: int,
    total_updates: int,
) -> None
```

## 引数

- `stop_reason`: 学習結果の停止理由。
- `completed_updates`: 反映まで成功した全データ更新数。
- `total_updates`: 要求した更新予算。

## 戻り値

なし。正常完了のときだけ返る。

## 使用場面

- 診断用の成果物を書いたあと、`condition_summary.csv` へ行を足す直前。
- `non_finite`、`out_of_memory`、更新数不足を完了済みにしたくないとき。

## 処理概要

1. `stop_reason` が `max_updates` で、成功更新数が `total_updates` と一致すれば返す。
2. それ以外は `RuntimeError`。ファイルは削除しない。

## 主なcontract / 注意事項

- 正常完了はこの2条件だけである。
- 実装上の例外を停止理由へ変換しない。この関数は、すでに返った停止理由を判定する。
- 呼び出し側は、この関数が返ったあとだけ集計行を返す。

## 関連API

[[90_src設計/05_Core_API_v1/training/reject_incomplete_scope_rows]]、[[90_src設計/05_Core_API_v1/training/train_scoped_full_batch_budget]]、[[90_src設計/05_Core_API_v1/training/run_resumable_condition_grid]]、[[90_src設計/05_Core_API_v1/Internal_API]]
