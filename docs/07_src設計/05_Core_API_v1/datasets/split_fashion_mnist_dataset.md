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

Dataset、各Subsetの長さ、split専用seed。

## 戻り値

`torch.utils.data.random_split()`が返すSubset群。

## 使用場面

train / Early-Stopping validation / rank-selection validationを再現可能に分離するとき。

## 処理の流れ（日本語）

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
