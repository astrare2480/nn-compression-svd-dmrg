# `make_fashion_mnist_loaders`

**Stability:** B  
**定義:** `src/nn_compression/datasets/fashion_mnist.py`

## 責務

Fashion-MNIST実験用のtrain/validation/test/train-eval DataLoaderを統一方針で作る。

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

Fashion-MNIST実験のDataLoader準備。

## ざっくりした処理

trainだけ`shuffle=True`で作成し、validation/test/train-eval/rank-validationは`shuffle=False`で作る。

## 主なcontract / 注意事項

評価のためにtrain用shuffle Generatorを余分に消費しない設計。

## 関連API

`non_shuffling_loader`, `make_torch_generator`
