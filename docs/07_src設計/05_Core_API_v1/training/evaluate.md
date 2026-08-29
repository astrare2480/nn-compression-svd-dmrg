# `evaluate`

**Stability:** A  
**定義:** `src/nn_compression/training/loops.py`

## 責務

Parameterを更新せず、loader全体のlossとaccuracyを評価する。評価中だけmodelをeval modeへ切り替え、処理後は呼び出し前のroot・全submoduleのtraining状態を復元する。

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

- `model`: 評価対象model。
- `loader`: `(images, labels)`を返す評価loader。
- `criterion`: loss関数。
- `device`: 評価device。

## 戻り値

`(avg_loss, accuracy)`。どちらもloader全体をsample単位で集計した値。

## 使用場面

validation/test評価、rank candidate評価、Early Stopping判定、Fine-tuning前後の比較。

## 処理の流れ（日本語）

1. **modelと全submoduleのtraining状態を保存する。**  
   rootだけではなく各submoduleの`.training`を個別に記録する。例えば「rootはtrainだがBatchNormだけeval」のような混在状態を壊さないため。
2. **一時的にmodelをeval modeへ切り替える。**  
   `model.eval()`を呼び、DropoutやBatchNormを推論時の挙動へする。
3. **gradient計算を無効化する。**  
   `torch.no_grad()`内で評価し、Parameter更新に不要なautograd graphを作らない。
4. **DataLoaderを最後まで走査する。**  
   各batchをdeviceへ移し、forwardとloss計算を行う。optimizerや`backward()`は呼ばない。
5. **lossと正解数をsample単位で集計する。**  
   batch平均lossをbatch sizeで重み付けし、dataset全体の平均lossになるよう蓄積する。
6. **空loaderを明示的に検出する。**  
   sample総数が0なら`ValueError`を送出する。
7. **平均lossとaccuracyを計算する。**  
   集計値をsample総数で割って結果を作る。
8. **処理前のtraining状態を必ず復元する。**  
   正常終了でも途中で例外が起きても、`finally`相当の処理で各moduleの`.training`を個別に元へ戻す。

### 処理フロー（短縮版）

```text
root + 全submoduleの状態保存
→ model.eval()
→ no_grad
→ loader全体をforward
→ loss / accuracy集計
→ empty確認
→ 結果作成
→ 全module状態を個別復元
```

## 主なcontract / 注意事項

- rootだけでなく**全submoduleのtraining状態を正常/例外時とも復元**する。
- `model.train(old_state)`で一括復元しない。再帰的変更によって混在状態が壊れるため。
- `len(loader)`を仮定しない。
- empty loaderは`ValueError`。

## 関連API

`train_one_epoch`, `fit_with_early_stopping`, `collect_compression_metrics`
