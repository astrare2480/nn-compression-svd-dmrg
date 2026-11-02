# `train_scoped_full_batch_budget`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_scoped_update.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

指定した scope だけを、共通の全データ更新予算まで学習する。初期状態も best 候補に含める。

## Signature

```python
train_scoped_full_batch_budget(
    model: nn.Module,
    source: DataLoader | GpuResidentTensorBatches,
    train_eval_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    scope: TTUpdateScope,
    layer_name: str,
    batch_size: int,
    total_updates: int,
    evaluation_interval: int,
    rank_cap: int,
    tt_learning_rate: float | None,
    dense_learning_rate: float | None,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    local_steps_per_pair: int | None = None,
    max_sweeps: int | None = None,
    center: int = 0,
    criterion: nn.Module | None = None,
) -> ScopedFullBatchBudgetResult
```

## 引数

- `model`: 学習を始めるモデル。関数内で更新される。
- `source`: 全データ勾配の供給源。
- `train_eval_loader`: 学習データ評価用の DataLoader。勾配更新には使わない。
- `validation_loader`: early-stopping validation。best 選択に使う。
- `scope`: 更新対象。
- `layer_name`: 対象 `TTLinear`。
- `batch_size`: 勾配集計の batch サイズ。
- `total_updates`: 全データ更新の予算。
- `evaluation_interval`: 評価する更新間隔。
- `rank_cap`: joint と tt-only の bond 上限。
- `tt_learning_rate`: TT pair の学習率。更新しない scope では `None`。
- `dense_learning_rate`: 通常 parameter の学習率。更新しない scope では `None`。
- `diagnostic_images`: 候補を反映する前の有限性確認に使う入力。
- `diagnostic_labels`: 診断 batch のラベル。
- `device`: 学習 device。
- `local_steps_per_pair`: 1つの pair を連続更新する回数。dense-only では `None`。
- `max_sweeps`: 左右1往復を繰り返す回数。dense-only では `None`。
- `center`: 開始時の正準中心。
- `criterion`: 平均損失。`None` なら平均 cross-entropy。

## 戻り値

`ScopedFullBatchBudgetResult`。要求した更新数まで到達したときだけ `stop_reason` は `max_updates`。

## 使用場面

- 同じ初期 checkpoint から、joint、tt-only、dense-only を独立に走らせるとき。
- 収束後の最適性能ではなく、共通の更新回数で曲線を比べるとき。

## 処理概要

1. 更新数 0 で train と early-stopping validation を評価し、初期状態を best 候補にする。
2. joint と tt-only は、右向きに pair を進めたあと左向きに戻る往復を、`local_steps_per_pair` 回ずつ繰り返す。評価間隔はその1往復の更新数と一致させる。
3. dense-only は通常 parameter の全データ更新だけを繰り返す。評価間隔は比較用の区切りであり、往復とは呼ばない。
4. 評価間隔ごとに履歴を追加し、validation loss が小さい方を best にする。同値なら累積更新数が小さい方を残す。
5. 予算まで到達すれば `max_updates`。非有限値は `non_finite`。CUDA のメモリ不足は `out_of_memory`。

## 主なcontract / 注意事項

- early stopping はしない。best の選択と停止判定は別。
- 学習率 0 では固定しない。除外する側の学習率は `None`。
- tt-only は通常 parameter の更新候補を作らない。dense-only は SVD も core 書き戻しも呼ばない。
- 対象 pair 以外の core が局所更新中に変わった場合は `RuntimeError`。停止理由へは変換しない。
- `NonFiniteLocalUpdate` と `torch.cuda.OutOfMemoryError` 以外の例外は伝播する。
- 学習時間と評価時間は別々に累積する。GPU 事前配置の時間は含まない。

## 関連API

[[90_src設計/05_Core_API_v1/training/TTUpdateScope]]、[[90_src設計/05_Core_API_v1/training/ScopedFullBatchBudgetResult]]、[[90_src設計/05_Core_API_v1/training/save_scoped_full_batch_run]]、[[90_src設計/05_Core_API_v1/training/evaluate]]、[[90_src設計/05_Core_API_v1/Internal_API]]
