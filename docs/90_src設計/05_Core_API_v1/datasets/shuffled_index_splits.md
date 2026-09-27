# `shuffled_index_splits`

**Stability:** B  
**定義:** `src/nn_compression/datasets/splits.py`

## 責務

`0..n-1`のindexをseed固定でshuffleし、指定した長さごとのindex listへ切り分ける。

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

- `n`: indexを作る対象Datasetの全sample数。候補indexは`0`から`n-1`。
- `lengths`: shuffle後のindex列を先頭から何件ずつ切り出すかを指定する各split長。
- `seed`: index shuffle専用Generatorへ設定するseed。同じ`n / lengths / seed`なら同じindex splitを再現できる。

## 戻り値

`lengths`と同じ順番で並ぶindex listのtuple。各listにはshuffle済みindexの対応区間が入り、`sum(lengths) < n`の場合は残りindexを返さない。

## 使用場面

train用と評価用でtransformが異なる別Dataset objectへ、**同じindex split**を適用したいとき。

## 処理概要

1. **`lengths`の合計を計算する。**
2. **合計が`n`を超えていないか確認する。**
3. **各lengthが0以上か確認する。**
4. **seed固定の専用`torch.Generator`を作る。**
5. **`torch.randperm(n, generator=...)`で全indexをランダム順に並べる。**
6. **`lengths`を先頭から順に読み、連続sliceで各splitを作る。**
7. **list群をtupleで返す。**  
   合計が`n`未満なら末尾の余ったindexは使わない。

## 主なcontract / 注意事項

- `sum(lengths) <= n`。
- 各lengthは0以上。
- 同じseedならindex順を再現できる。

## 関連API

`split_fashion_mnist_dataset`
