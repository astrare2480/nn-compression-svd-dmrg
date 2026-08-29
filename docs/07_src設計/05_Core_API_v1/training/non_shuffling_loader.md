# `non_shuffling_loader`

**Stability:** B  
**定義:** `src/nn_compression/training/fit.py`

## 責務

元DataLoaderと同じDatasetを、shuffleせず評価するためのDataLoaderを作る。

## Signature

```python
non_shuffling_loader(loader: DataLoader) -> DataLoader
```

## 引数

`loader`: 学習等で使っている元DataLoader。

## 戻り値

同Dataset・主なloader設定を引き継いだ`shuffle=False` DataLoader。

## 使用場面

train metricを再評価したいが、学習用shuffle Generatorを進めたくないとき。

## ざっくりした処理

元loaderからdataset/batch_size/worker設定等を取り出し、新しいshuffleなしloaderを構築する。

## 主なcontract / 注意事項

custom sampler / custom batch_sampler / DataLoader subclassの完全cloneを保証するAPIではない。

## 関連API

`fit_with_early_stopping`, `make_fashion_mnist_loaders`
