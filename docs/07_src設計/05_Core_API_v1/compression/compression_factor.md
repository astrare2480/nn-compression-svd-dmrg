# `compression_factor`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

元TensorがTucker表現の何倍の要素数を持つか、圧縮倍率として返す。

## Signature

```python
compression_factor(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> float
```

## 引数

`shape`, `{mode: rank}`。

## 戻り値

```text
original_element_count / tucker_parameter_count
```

大きいほど強い圧縮。

## 使用場面

「何倍小さくなったか」でrank候補を比較したいとき。

## ざっくりした処理

元要素数を計算し、Tucker表現要素数で割る。

## 主なcontract / 注意事項

parameter ratioの逆数方向の指標。

## 関連API

`tucker_parameter_count`, `parameter_ratio`
