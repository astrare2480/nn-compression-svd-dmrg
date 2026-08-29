# `conv2d_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

1 sampleがConv2dを通る理論MACsを計算する。

## Signature

```python
conv2d_macs(
    conv,
    out_h: int,
    out_w: int,
) -> int
```

## 引数

- `conv`: Conv2d layer。
- `out_h`, `out_w`: 出力feature mapの空間size。

## 戻り値

```text
out_h * out_w * out_ch * in_ch_per_group * kH * kW
```

## 使用場面

baseline Convの理論計算量を数えるとき。

## ざっくりした処理

出力位置数 × 出力channel数 × 1出力に必要なkernel積和数を掛ける。

## 主なcontract / 注意事項

- `out_h/out_w`はbool/Tensor scalar以外の正整数。
- `weight.shape[1]`はgrouped Convで既に`in_channels/groups`なので、さらにgroupsで割らない。

## 関連API

`compressed_conv2d_macs`, `estimate_conv2d_macs`
