# `ScopedFullBatchBudgetResult`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

固定した全データ更新予算の1条件について、履歴、best、最終 state、停止理由を保持する。

## Signature

```python
ScopedFullBatchBudgetResult(
    scope: str,
    stop_reason: str,
    stop_detail: str,
    completed_updates: int,
    requested_updates: int,
    history: list,
    local_updates: list,
    best_update_count: int,
    best_validation_loss: float,
    best_state: dict,
    initial_state: dict,
    center: int | None,
    seconds_train: float,
    seconds_eval: float,
    svd_call_count: int,
    dense_candidate_count: int,
    core_writeback_count: int,
) -> None
```

## 引数

- `scope`: `"joint"`、`"tt_only"`、`"dense_only"`。
- `stop_reason`: `max_updates`、`non_finite`、`out_of_memory` のいずれか。
- `stop_detail`: 停止時のメッセージ。正常完了では空文字。
- `completed_updates`: 反映まで成功した全データ更新の数。
- `requested_updates`: 要求した更新予算。
- `history`: 更新数 0 と、評価間隔ごとの評価行。
- `local_updates`: 成功した各全データ更新の行。該当しない項目は `None`。
- `best_update_count`: best の累積更新数。初期状態は 0。
- `best_validation_loss`: best の early-stopping validation loss。
- `best_state`: best 時点の CPU state。
- `initial_state`: 更新前の CPU state。
- `center`: pair 更新後の正準中心。dense-only は `None`。
- `seconds_train`: 更新に使った累積秒。評価時間は含まない。
- `seconds_eval`: 評価に使った累積秒。
- `svd_call_count`: 方向付き SVD を呼んだ回数。
- `dense_candidate_count`: 通常 parameter の更新候補を作った回数。
- `core_writeback_count`: 更新した TT core をモデルへ書き戻した回数。

## 戻り値

インスタンス。

## 使用場面

- 3,200 更新のような共通予算で、scope ごとの学習曲線を残すとき。
- best と final を別々に保存するとき。

## 処理概要

1. 学習関数が、成功した更新と評価点だけをこのオブジェクトへ集める。
2. 保存関数が、初期・最終・best と CSV、JSON へ書き出す。

## 主なcontract / 注意事項

- `max_updates` は要求した更新数へ到達したときだけ。
- best は early-stopping validation loss の最小。同値なら累積更新数が小さい方。
- best の accuracy は、その best 行の accuracy であり、期間中の最大 accuracy ではない。
- dense-only の pair、中心、retained rank、切り捨て量、TT 勾配は `None`。数値 0 は入れない。
- tt-only の通常 parameter 勾配と相対更新量は `None`。

## 関連API

[[90_src設計/05_Core_API_v1/training/train_scoped_full_batch_budget]]、[[90_src設計/05_Core_API_v1/training/save_scoped_full_batch_run]]、[[90_src設計/05_Core_API_v1/Internal_API]]
