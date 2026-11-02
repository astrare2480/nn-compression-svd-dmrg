# `TTLinearFullBatchSweepResult`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

全データ勾配スイープの停止理由、完了往復数、best state、履歴をまとめる。`best_state`と`initial_state`は実モデルとstorageを共有しない複製である。

## Signature

```python
TTLinearFullBatchSweepResult(
    stop_reason: str,
    stop_detail: str,
    completed_sweeps: int,
    successful_local_updates: int,
    incomplete_sweep_successes: int,
    best_sweep: int,
    best_validation_loss: float,
    best_state: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    center: int,
    last_success_pair: int | None,
    last_success_direction: str | None,
    history: list[dict],
    local_updates: list[dict],
    stop_sweep: int | None,
    stop_pair: int | None,
    stop_direction: str | None,
    stop_local_step: int | None,
) -> None
```

## 引数

- `stop_reason`: `max_sweeps`、`early_stopping`、`non_finite`、`svd_failure`のいずれか。`early_stopping` は `early_stopping_enabled=True` のときだけ選ばれる。`ValueError`、`AssertionError`、その他の例外は結果へ変換せず呼び出し元へ返す。
- `stop_detail`: 停止理由の説明。最大往復やearly stoppingは数学的収束ではない。
- `completed_sweeps`: 左右の往復が最後まで終わり、評価まで終えた回数。途中失敗の往復は含めない。
- `successful_local_updates`: 実モデルへ反映した局所更新の累計。途中往復の成功も含む。
- `incomplete_sweep_successes`: 完了しなかった往復の中で反映済みの回数。正常終了時は0。
- `best_sweep`: validation lossが最小だったsweep。sweep 0を含む。
- `best_validation_loss`: その最小値。`min_delta`とは独立。
- `best_state`: best時点の`state_dict`の複製。
- `initial_state`: 学習開始時点の`state_dict`の複製。
- `center`: 最後に成功した更新のあとにある直交中心。
- `last_success_pair`: 最後に成功したpair。一度も成功していなければ`None`。
- `last_success_direction`: その向き。
- `history`: 完了したsweepとsweep 0の評価履歴。
- `local_updates`: 局所更新と、失敗時の1行。
- `stop_sweep`: 失敗した往復の番号。正常終了時は`None`。
- `stop_pair`: 失敗したpair。
- `stop_direction`: 失敗した向き。
- `stop_local_step`: 失敗したpair内step。

## 戻り値

上記フィールドを持つfrozen dataclass。

## 使用場面

- `train_tt_linear_full_batch_sweeps`の戻り値として、保存と再読み込みの材料にするとき。

## 処理概要

1. 停止理由と、完了往復・成功更新・途中位置を分けて持つ。
2. bestと初期stateはdeepcopyしたdictとして持つ。

## 主なcontract / 注意事項

- 途中失敗のモデルは、最後の成功更新までを保持した実モデル側にある。bestには未評価の途中状態を入れない。
- `stop_reason="max_sweeps"`も`"early_stopping"`も、収束したことの証明ではない。

## 関連API

[[90_src設計/05_Core_API_v1/training/train_tt_linear_full_batch_sweeps]]、[[90_src設計/05_Core_API_v1/training/save_tt_linear_full_batch_sweep_run]]、[[90_src設計/05_Core_API_v1/Internal_API]]
