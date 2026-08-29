# `evaluate`

**Stability:** A  
**定義:** `src/nn_compression/training/loops.py`

## 責務

Parameterを更新せず、loader全体のlossとaccuracyを評価する。

## Signature

```python
evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]
```

## 引数

model、評価loader、criterion、device。

## 戻り値

`(avg_loss, accuracy)`。

## 使用場面

validation/test評価、rank candidate評価、Early Stopping判定。

## ざっくりした処理

```text
全moduleのtraining状態を保存
→ model.eval()
→ torch.no_grad()
→ loader全体forward
→ sample-weighted loss/accuracy集計
→ finallyで各moduleの状態を個別復元
```

## 主なcontract / 注意事項

- rootだけでなく**全submoduleのtraining状態を正常/例外時とも復元**。
- `len(loader)`を仮定しない。
- empty loaderは`ValueError`。

## 関連API

`train_one_epoch`, `fit_with_early_stopping`, `collect_compression_metrics`
