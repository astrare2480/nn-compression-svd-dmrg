# `train_one_epoch`

**Stability:** A  
**定義:** `src/nn_compression/training/loops.py`

## 責務

DataLoaderを1回走査し、順伝播・逆伝播・optimizer更新を行う1 epoch分の学習処理。

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
- `optimizer`: Parameter更新に使うoptimizer。
- `device`: 学習を実行するdevice。

## 戻り値

`(avg_loss, accuracy)`。

- `avg_loss`: sample数で加重したepoch全体の平均loss。
- `accuracy`: epoch中に処理した全sampleに対する正解率。

## 使用場面

baseline modelの通常学習、SVD/Tucker圧縮後のFine-tuning。

## 処理の流れ（日本語）

1. **modelを学習モードへ切り替える。**  
   `model.train()`を呼び、DropoutやBatchNormを学習時の挙動へする。この関数は学習APIなので、呼び出し前のtrain/eval状態を復元する設計ではない。
2. **epoch集計用の変数を初期化する。**  
   loss合計、正解数、sample総数を0から開始する。
3. **DataLoaderを1 batchずつ走査する。**  
   画像とラベルを指定deviceへ移動する。
4. **前回のgradientをクリアする。**  
   `optimizer.zero_grad(set_to_none=True)` を呼び、今回のbatch用にgradient状態を初期化する。
5. **forwardとloss計算を行う。**  
   `outputs = model(images)` のあと、`criterion(outputs, labels)` でlossを得る。
6. **逆伝播してParameterを更新する。**  
   `loss.backward()` でgradientを計算し、`optimizer.step()` でParameterを更新する。
7. **lossと正解数をsample単位で集計する。**  
   batch平均lossへbatch sizeを掛けて加算することで、最終的にsample-weighted averageを計算できるようにする。予測classは`argmax`で求める。
8. **空loaderを明示的に検出する。**  
   走査後のsample総数が0なら、偶発的な0除算ではなく`ValueError`を送出する。
9. **epoch平均を計算して返す。**  
   loss合計をsample総数で割り、正解数もsample総数で割って`(avg_loss, accuracy)`を返す。

### 処理フロー（短縮版）

```text
model.train()
→ batchをdeviceへ移動
→ zero_grad
→ forward
→ loss
→ backward
→ optimizer.step
→ loss / 正解数を集計
→ empty確認
→ epoch平均を返す
```

## 主なcontract / 注意事項

- `len(loader)`を仮定しないため、長さを持たないIterableDatasetでも実走査できる。
- 実走査後に0 sampleなら明示的`ValueError`。
- 学習APIなので呼出し後modelはtrain modeになる。

## 関連API

`evaluate`, `fit_with_early_stopping`
