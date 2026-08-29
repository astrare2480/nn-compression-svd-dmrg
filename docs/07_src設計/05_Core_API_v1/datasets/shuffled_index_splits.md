# `shuffled_index_splits`

**Stability:** B  
**定義:** `src/nn_compression/datasets/splits.py`

## 責務

`0..n-1`のindexをseed固定でshuffleし、指定長のindex listへ分ける。

## Signature

```python
shuffled_index_splits(
    n: int,
    lengths: tuple[int, ...],
    *,
    seed: int,
) -> tuple[list[int], ...]
```

## 引数

全件数`n`、各split長、seed。

## 戻り値

指定長ごとのindex list tuple。

## 使用場面

train用/評価用でtransformの異なるDatasetへ**同じindex split**を適用したいとき。

## ざっくりした処理

専用Generator → `torch.randperm(n)` → lengthsに従って連続slice。

## 主なcontract / 注意事項

- `sum(lengths) <= n`。
- 各lengthは0以上。
- 余ったindexは使わない。

## 関連API

`split_fashion_mnist_dataset`
