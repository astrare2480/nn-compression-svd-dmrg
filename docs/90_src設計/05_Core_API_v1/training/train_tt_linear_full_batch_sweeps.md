# `train_tt_linear_full_batch_sweeps`

**Stability:** A  
**定義:** `src/nn_compression/training/tt_linear_full_batch_sweep.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/00_full_batch_two_core_sweep_training.ipynb`

## 責務

sweep 0を評価したあと、右向きスイープと左向きスイープを1往復として最大`max_sweeps`回繰り返し、validation lossでbestとearly stoppingを別々に管理する。

## Signature

```python
train_tt_linear_full_batch_sweeps(
    model: nn.Module,
    gradient_loader: DataLoader,
    train_eval_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    layer_name: str,
    rank_cap: int,
    tt_pair_learning_rate: float,
    dense_parameter_learning_rate: float,
    max_sweeps: int,
    local_steps_per_pair: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    diagnostic_images: torch.Tensor,
    diagnostic_labels: torch.Tensor,
    device: torch.device,
    center: int = 0,
    criterion: nn.Module | None = None,
    observer: object | None = None,
    early_stopping_enabled: bool = True,
) -> TTLinearFullBatchSweepResult
```

## 引数

- `model`: 初期正準化済みのモデル。この関数は毎pairで正準化し直さない。
- `gradient_loader`: 全データ勾配用。`shuffle=False`。
- `train_eval_loader`: 往復後の学習指標用。`shuffle=False`。勾配集計の損失は再利用しない。
- `validation_loader`: early stoppingとbest用。`shuffle=False`。
- `layer_name`: 対象`TTLinear`。
- `rank_cap`: bond上限。
- `tt_pair_learning_rate`: `T`の学習率。
- `dense_parameter_learning_rate`: 通常parameterの学習率。
- `max_sweeps`: 最大往復数。epoch数ではない。0ならsweep 0だけ評価する。
- `local_steps_per_pair`: 1 pairあたりの局所更新回数。
- `early_stopping_patience`: 改善しない往復がこの回数続いたら停止する。
- `early_stopping_min_delta`: 基準からこの値を超えてvalidation lossが下がったときだけ改善と数える。
- `diagnostic_images`: 候補の有限性確認用の固定batch。
- `diagnostic_labels`: そのラベル。
- `device`: 計算device。
- `center`: 開始時の直交中心。既定は0。
- `criterion`: 平均損失。`None`なら平均cross-entropy。
- `observer`: 各局所更新の直前に呼ばれる任意の関数。
- `early_stopping_enabled`: 既定 `True` では、非改善が `early_stopping_patience` 続いたら止まる。`False` ではその停止をしない。best の更新はどちらでも行う。巨大な patience で無効化したことにはしない。

## 戻り値

`TTLinearFullBatchSweepResult`。実モデルは最後に成功した局所更新までの状態で残る。

## 使用場面

- Fashion-MNIST TT-MLPで、全データ勾配の左右スイープをearly stoppingまで実行するとき。

## 処理概要

1. sweep 0を学習データとvalidationで評価し、そのvalidation lossをearly-stoppingの基準かつ最初のbestにする。
2. 各往復で右向きスイープのあと左向きスイープを行う。6 core・局所step 1なら10回の全データ勾配になる。
3. 往復が終わってから、固定したモデルを通常forwardで評価する。最後のpairの集計損失は評価に使わない。
4. `baseline - validation_loss > min_delta`のときだけ基準を更新して非改善回数を0にする。それ以外は回数を1増やす。
5. bestは`min_delta`と無関係に、厳密に小さいvalidation lossのときだけ更新する。
6. `early_stopping_enabled` が真なら patience に達した時点で止まる。偽なら非改善では止まらず、最大往復まで進む。数値失敗ではどちらも止まる。

## 主なcontract / 注意事項

- minibatchごとにparameterは更新しない。
- 通常parameterは各局所更新で更新される。TT coreだけの実験ではない。
- 途中失敗の往復は完了数に入れない。未評価の途中状態はbestにしない。
- `max_sweeps`到達とearly stoppingは、数学的収束の証明ではない。損失の単調減少は要求しない。
- `early_stopping_enabled=False` でも best は往復終了時の validation loss の厳密最小で更新する。`min_delta` 未満の改善も best にはなる。非改善回数は履歴に残るが、停止には使わない。
- 制御された停止は`NonFiniteLocalUpdate`（`non_finite`）と`torch.linalg.LinAlgError`（`svd_failure`）だけである。失敗した候補は実モデルへ反映しない。
- `ValueError`、`AssertionError`、`RuntimeError`を含む想定外の例外は捕捉しない。保存処理へ進まず、呼び出し元へそのまま伝播する。

## 関連API

[[90_src設計/05_Core_API_v1/training/sweep_tt_linear_pairs_left_to_right]]、[[90_src設計/05_Core_API_v1/training/sweep_tt_linear_pairs_right_to_left]]、[[90_src設計/05_Core_API_v1/training/evaluate]]、[[90_src設計/05_Core_API_v1/training/TTLinearFullBatchSweepResult]]、[[90_src設計/05_Core_API_v1/Internal_API]]
