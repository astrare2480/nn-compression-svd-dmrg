# `compressed_linear_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

`Linear(in→rank) → Linear(rank→out)` の理論MACsを計算する。

## Signature

```python
compressed_linear_macs(
    in_features: int,
    out_features: int,
    rank: int,
) -> int
```

## 引数

入力dimension、出力dimension、SVD rank。

## 戻り値

```text
in_features * rank + rank * out_features
```

## 使用場面

Linear SVDのrank候補を理論計算量で比較するとき。

## ざっくりした処理

rankを分解可能上限でvalidationし、2つのLinear MACsを加算する。

## 主なcontract / 注意事項

rankはbool/Tensor scalar不可、`1 <= rank <= min(in_features, out_features)`。

## 関連API

`linear_macs`, `factorize_linear_layer`
