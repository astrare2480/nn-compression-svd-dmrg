# `split_fashion_mnist_dataset`

**Stability:** B  
**定義:** `src/nn_compression/datasets/fashion_mnist.py`

## 責務

Fashion-MNIST学習Datasetを、seed固定で再現可能な複数Subsetへ分割する。

## Signature

```python
split_fashion_mnist_dataset(
    dataset: Dataset,
    split_lengths: tuple[int, ...],
    *,
    seed: int,
)
```

## 引数

Dataset、各Subsetの長さ、split seed。

## 戻り値

`torch.utils.data.random_split()`が返すSubset群。

## 使用場面

train / Early-Stopping validation / rank-selection validationを分離するとき。

## ざっくりした処理

`make_torch_generator(seed)` → `random_split(dataset, lengths, generator=...)`。

## 主なcontract / 注意事項

split乱数と学習DataLoader shuffle乱数を別Generatorとして扱う。

## 関連API

`make_torch_generator`, `shuffled_index_splits`
