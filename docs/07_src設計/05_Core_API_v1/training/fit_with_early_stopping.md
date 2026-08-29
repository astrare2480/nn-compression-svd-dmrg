# `fit_with_early_stopping`

**Stability:** B  
**定義:** `src/nn_compression/training/fit.py`

## 責務

validation loss基準のEarly Stopping付き学習を行い、最良epochのstateへmodelを戻す。

## Signature

```python
fit_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    max_epochs,
    patience,
    min_delta,
    *,
    reevaluate_train=True,
    log_every_epoch=False,
    train_eval_loader=None,
)
```

## 引数

- `max_epochs`: 最大epoch数。
- `patience`: 改善なしを許す連続epoch数。
- `min_delta`: 改善とみなす最小validation loss差。
- `reevaluate_train`: train metricをshuffleなしloaderで再評価するか。
- `train_eval_loader`: 明示的なtrain評価loader。

## 戻り値

次を含むdict。

```text
model
best_epoch
best_validation_loss
train_loss_history
validation_loss_history
history
```

## 使用場面

baseline学習、圧縮後Fine-tuningを共通条件で行うとき。

## ざっくりした処理

```text
train評価loader準備
→ epochごとにtrain_one_epoch
→ evaluate(val)
→ 必要ならshuffleなしでtrain再評価
→ validation loss改善ならstate_dictをdeepcopy
→ patience到達で停止
→ best stateをload
```

## 主なcontract / 注意事項

- shuffle付きtrain loaderをmetric再評価で余分に走査しない。
- `train_eval_loader`指定時は`reevaluate_train=True`が必要。
- scalar validationはHOOI/benchmarkほど全面統一されていない既知制約。

## 関連API

`train_one_epoch`, `evaluate`, `non_shuffling_loader`
