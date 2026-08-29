# `train_one_epoch`

**Stability:** A  
**定義:** `src/nn_compression/training/loops.py`

## 責務

DataLoaderを1回走査し、順伝播・逆伝播・optimizer更新を行う1 epoch学習処理。

## Signature

```python
train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]
```

## 引数

- `model`: 学習対象model。
- `train_loader`: `(images, labels)`を返す学習loader。
- `criterion`: loss関数。
- `optimizer`: optimizer。
- `device`: 学習device。

## 戻り値

`(avg_loss, accuracy)`。lossはサンプル数加重平均、accuracyは全サンプル正解率。

## 使用場面

baseline学習、SVD/Tucker圧縮後のFine-tuning。

## ざっくりした処理

```text
model.train()
→ 各batchをdeviceへ
→ zero_grad
→ forward
→ loss.backward
→ optimizer.step
→ loss/正解数を集計
```

## 主なcontract / 注意事項

- `len(loader)`を仮定しない。
- 実走査後に0 sampleなら明示的`ValueError`。
- 学習APIなので呼出し後modelはtrain modeになる。

## 関連API

`evaluate`, `fit_with_early_stopping`
