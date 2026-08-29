# `make_fashion_mnist_loaders`

**Stability:** B  
**定義:** `src/nn_compression/datasets/fashion_mnist.py`

## 責務

Fashion-MNIST実験用のtrain / validation / test / train-eval、およびoptional rank-validation DataLoaderを統一したshuffle方針で作る。

## Signature

```python
make_fashion_mnist_loaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    *,
    train_batch_size,
    validation_batch_size,
    test_batch_size,
    rank_validation_dataset=None,
    train_generator=None,
)
```

## 引数

各Dataset、batch size、optional rank-selection Dataset、train shuffle用Generator。

## 戻り値

```text
{
  "train_loader": ...,
  "validation_loader": ...,
  "test_loader": ...,
  "train_eval_loader": ...,
  ["validation_loader_rank": ...]
}
```

## 使用場面

Fashion-MNIST実験で、学習用のランダム順序と評価用の固定順序を明確に分けたDataLoader群を準備するとき。

## 処理の流れ（日本語）

1. **train loaderを作る。**  
   `shuffle=True`とし、指定された`train_generator`を渡す。これだけが学習順をランダム化するloader。
2. **validation loaderを作る。**  
   `shuffle=False`で固定順序にする。
3. **test loaderを同様に`shuffle=False`で作る。**
4. **train-eval loaderを作る。**  
   Datasetはtrainと同じだが`shuffle=False`にし、学習用Generatorを進めずtrain metricを再評価できるようにする。
5. **rank-validation Datasetが指定されていれば専用loaderを追加する。**  
   これもcandidate比較用なので`shuffle=False`。
6. **名前付きdictとして返す。**

### shuffle方針

```text
train                 → shuffle=True
validation            → shuffle=False
test                  → shuffle=False
train_eval            → shuffle=False
rank_validation       → shuffle=False
```

## 主なcontract / 注意事項

評価のためにtrain用shuffle Generatorを余分に消費しない設計。

## 関連API

`non_shuffling_loader`, `make_torch_generator`
