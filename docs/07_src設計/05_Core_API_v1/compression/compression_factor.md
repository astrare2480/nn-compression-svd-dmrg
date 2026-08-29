# `compression_factor`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

元Tensor要素数がTucker表現要素数の何倍あるかを圧縮倍率として返す。値が大きいほどTucker表現が小さい。

## Signature

```python
compression_factor(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> float
```

## 引数

- `shape`: 圧縮前Tensorのshape。元Tensorの総要素数と各modeのdimensionを決める。
- `ranks`: `{mode: rank}`形式のTucker rank指定。keyは圧縮するmode、valueはそのmodeに残すrankで、未指定modeは元dimensionをcoreに残す。

## 戻り値

元TensorがTucker表現の何倍の要素数を持つかを表す圧縮倍率の`float`。

```text
original_element_count / tucker_parameter_count(shape, ranks)
```

`1.0`なら要素数は同じ、`2.0`ならTucker表現は元の約1/2、値が大きいほど強く圧縮されている。

## 使用場面

「元Tensorに対して何倍小さい表現か」という倍率でrank候補を比較するとき。

## 処理概要

1. **shapeとranksを検証する。**
2. **元Tensorの総要素数をshapeの積で求める。**
3. **`tucker_parameter_count()`でcore + factorsの要素数を求める。**
4. **元要素数をTucker要素数で割り、圧縮倍率を返す。**

## 主なcontract / 注意事項

`parameter_ratio()`とは分子・分母が逆の指標。両方を同時に報告する場合は意味を混同しない。

## 関連API

`tucker_parameter_count`, `parameter_ratio`
