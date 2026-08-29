# `split_fashion_mnist_dataset`

**Stability:** B  
**定義:** `src/nn_compression/datasets/fashion_mnist.py`

## 責務

Fashion-MNIST学習Datasetを、指定lengthとseedに従って再現可能な複数Subsetへ分割する。

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

- `dataset`: 分割元となるDataset。通常はFashion-MNISTのfull train split。
- `split_lengths`: 返したい各Subsetのsample数を順番に並べたtuple。各値がどの用途かは呼び出し側の並びで決まる。
- `seed`: `random_split()`専用Generatorへ設定するseed。学習DataLoaderのshuffle RNGとは分離してsplit indexを再現するために使う。

## 戻り値

`split_lengths`と同じ順序で並ぶ`torch.utils.data.Subset`群。各Subsetは元`dataset`を共有しつつ、それぞれ割り当てられたindexだけを参照する。

## 使用場面

train / Early-Stopping validation / rank-selection validationを再現可能に分離するとき。

## 処理概要

1. **split専用の`torch.Generator`を作る。**  
   `make_torch_generator(seed)`を使い、global RNGや学習shuffle用Generatorと分ける。
2. **Datasetと`split_lengths`を`random_split()`へ渡す。**
3. **専用Generatorでindex割当を決める。**  
   同じseed・同じDataset長・同じsplit lengthsなら同じ分割を再現できる。
4. **生成されたSubset群をそのまま返す。**

## 主なcontract / 注意事項

split乱数とtraining DataLoaderのshuffle乱数を同じGeneratorで共有しない。

## 関連API

`make_torch_generator`, `shuffled_index_splits`
