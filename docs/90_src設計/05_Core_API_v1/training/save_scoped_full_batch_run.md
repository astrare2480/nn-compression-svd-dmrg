# `save_scoped_full_batch_run`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

1条件の初期モデル、最終モデル、best、履歴、局所ログ、実行状態、設定を保存する。

## Signature

```python
save_scoped_full_batch_run(
    directory: Path | str,
    model: nn.Module,
    result: ScopedFullBatchBudgetResult,
    config: Mapping,
) -> None
```

## 引数

- `directory`: 条件ごとの保存先。無ければ作る。既存ディレクトリは削除しない。
- `model`: 最後に成功した更新後のモデル。final として保存する。
- `result`: 学習結果。初期 state と best state を別々に持つ。
- `config`: `layer_name`、`out_modes`、`in_modes` を含む設定。

## 戻り値

なし。

## 使用場面

- scope ごとの `initial_model.pt`、`final_model.pt`、`best_model.pt` を残すとき。
- 再開判定が読む `run_state.json` と `config.json` を残すとき。

## 処理概要

1. 同じディレクトリの一時ファイルへ書いてから、本番ファイルへ置換する。
2. `history.csv` と `local_updates.csv` では、`None` を空欄にする。
3. `run_state.json` と `config.json` に設定と停止理由を書く。

## 主なcontract / 注意事項

- best と final は別ファイル。final は best で上書きしない。
- 空欄は、その項目を計算しなかったことを表す。切り捨て量が 0 だった記録とは区別する。
- 途中のディレクトリを完成済みとして削除しない。
- 読み戻しは既存の `load_tt_linear_sweep_checkpoint` を使える。

## 関連API

[[90_src設計/05_Core_API_v1/training/ScopedFullBatchBudgetResult]]、[[90_src設計/05_Core_API_v1/training/load_tt_linear_sweep_checkpoint]]、[[90_src設計/05_Core_API_v1/training/run_resumable_condition_grid]]、[[90_src設計/05_Core_API_v1/Internal_API]]
