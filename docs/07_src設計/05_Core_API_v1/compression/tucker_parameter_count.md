# `tucker_parameter_count`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

指定rankのTucker表現を保存するためのcore + factor総要素数を計算する。

## Signature

```python
tucker_parameter_count(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> int
```

## 引数

- `shape`: 元Tensor shape。
- `ranks`: `{mode: rank}`。未指定modeはcoreに元dimensionを残す。

## 戻り値

```text
core要素数 + Σ(I_mode * rank_mode)
```

の整数。

## 使用場面

Tucker rank候補の理論parameter量を比較するとき。

## ざっくりした処理

```text
shape/rank validation
→ core shapeの要素積
→ 各factor要素数を加算
```

## 主なcontract / 注意事項

モデル全体のParameter数ではなく、1 TensorのTucker表現要素数。

## 関連API

`parameter_ratio`, `compression_factor`
