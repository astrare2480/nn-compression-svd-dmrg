# `parameter_ratio`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

Tucker表現の要素数が元Tensor要素数の何倍かを比率で返す。値が小さいほど保存量が小さい。

## Signature

```python
parameter_ratio(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> float
```

## 引数

`shape`と`{mode: rank}`。

## 戻り値

```text
tucker_parameter_count(shape, ranks) / original_element_count
```

## 使用場面

rank sweepでTucker表現の圧縮の強さを「元の何割か」で比較するとき。

## 処理の流れ（日本語）

1. **元shapeが正のdimensionだけで構成されているか検証する。**
2. **ranksをTucker共通contractで検証する。**
3. **元Tensorの総要素数をshapeの積で計算する。**
4. **`tucker_parameter_count()`でTucker表現の総要素数を計算する。**
5. **Tucker要素数を元要素数で割って返す。**

### 処理フロー（短縮版）

```text
shape / ranks検証
→ original element count
→ tucker_parameter_count
→ Tucker / original
```

## 主なcontract / 注意事項

`ranks={}`ではidentity表現となり、factorがなくcoreが元Tensorそのものなのでratioは1.0相当になる。

## 関連API

`tucker_parameter_count`, `compression_factor`
