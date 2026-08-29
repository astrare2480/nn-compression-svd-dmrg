# `conv2d_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

1 sampleがConv2dを通るときの理論MACsを、出力feature map sizeとweight shapeから計算する。

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
- `out_h`, `out_w`: このConvの出力feature mapの高さ・幅。

## 戻り値

```text
out_h * out_w * out_ch * in_ch_per_group * kH * kW
```

## 使用場面

baseline Convの理論計算量を数えるとき。

## 処理の流れ（日本語）

1. **`out_h/out_w`を正整数として検証する。**  
   boolやTensor scalarをサイズとして受け入れない。
2. **Conv weight shapeを読む。**  
   `(out_ch, in_ch_per_group, kH, kW)`を取得する。
3. **1つの出力位置・1出力channelに必要な積和数を求める。**  
   `in_ch_per_group * kH * kW`。
4. **出力位置数`out_h*out_w`と出力channel数`out_ch`を掛ける。**
5. **1 sampleあたり理論MACsを返す。**

## 主なcontract / 注意事項

- `out_h/out_w`はbool/Tensor scalar以外の正整数。
- grouped Convでは`weight.shape[1]`がすでに`in_channels/groups`なので、さらに`groups`で割らない。

## 関連API

`compressed_conv2d_macs`, `estimate_conv2d_macs`
