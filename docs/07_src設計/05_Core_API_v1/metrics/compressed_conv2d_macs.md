# `compressed_conv2d_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

SVDで2層化したConv2dの理論MACsを計算する。

## Signature

```python
compressed_conv2d_macs(
    conv,
    rank: int,
    out_h: int,
    out_w: int,
) -> int
```

## 引数

元Conv、SVD rank、出力空間size。

## 戻り値

```text
out_h*out_w*rank*in_ch*kH*kW
+
out_h*out_w*out_ch*rank
```

## 使用場面

Conv SVD candidateの理論計算量比較。

## ざっくりした処理

`factorize_conv2d_layer()`の空間Convと1x1 Convを別々に数えて加算する。

## 主なcontract / 注意事項

- compressed Convは`groups=1`のみ。
- rankはbool/Tensor scalar不可、分解可能上限以内。
- `out_h/out_w`は正整数。

## 関連API

`conv2d_macs`, `estimate_conv2d_macs`, `factorize_conv2d_layer`
