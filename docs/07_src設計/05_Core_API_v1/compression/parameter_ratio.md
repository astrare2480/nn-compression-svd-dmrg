# `parameter_ratio`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

Tucker表現の要素数が元Tensorの何倍かを比率で返す。

## Signature

```python
parameter_ratio(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> float
```

## 引数

`shape`, `{mode: rank}`。

## 戻り値

```text
tucker_parameter_count / original_element_count
```

小さいほど圧縮されている。

## 使用場面

rank sweepで圧縮の強さを比較するとき。

## ざっくりした処理

shape/ranksを検証し、元要素数とTucker要素数の比を取る。

## 主なcontract / 注意事項

`ranks={}` ではidentity表現としてratioは1.0相当になる。

## 関連API

`tucker_parameter_count`, `compression_factor`
