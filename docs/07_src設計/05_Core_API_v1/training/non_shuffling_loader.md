# `non_shuffling_loader`

**Stability:** B  
**定義:** `src/nn_compression/training/fit.py`

## 責務

元DataLoaderと同じDatasetを、shuffleせず評価するための新しいDataLoaderとして組み直す。

## Signature

```python
non_shuffling_loader(loader: DataLoader) -> DataLoader
```

## 引数

`loader`: 学習等で使っている元DataLoader。

## 戻り値

同Dataset・主なloader設定を引き継いだ`shuffle=False` DataLoader。

## 使用場面

train metricを再評価したいが、学習用shuffle Generatorを余分に進めたくないとき。

## 処理の流れ（日本語）

1. **元loaderからDatasetと基本設定を読み取る。**  
   `dataset`, `batch_size`, `num_workers`, `collate_fn`, `pin_memory`, `timeout`など、評価用loaderでも維持したい設定を取得する。
2. **shuffleを明示的に無効化する。**  
   新loaderでは`shuffle=False`とし、学習用のランダム順序やGeneratorを評価で消費しないようにする。
3. **評価用途なので`drop_last=False`にする。**  
   最後の不完全batchも含め、Dataset全体を評価対象にする。
4. **workerを使う場合だけworker関連設定を引き継ぐ。**  
   `num_workers > 0`なら`worker_init_fn`, `multiprocessing_context`, `prefetch_factor`, `persistent_workers`を引き継ぐ。
5. **新しいDataLoaderを構築して返す。**  
   元loader自体は変更せず、評価専用の別インスタンスを返す。

### 処理フロー（短縮版）

```text
元DataLoader
→ Dataset / batch / worker設定を取得
→ shuffle=False, drop_last=Falseへ固定
→ worker設定を必要に応じて継承
→ 評価用DataLoaderを新規作成
```

## 主なcontract / 注意事項

- custom sampler / custom batch_sampler / DataLoader subclassの完全cloneを保証するAPIではない。
- 学習loaderそのものを変更しない。

## 関連API

`fit_with_early_stopping`, `make_fashion_mnist_loaders`
