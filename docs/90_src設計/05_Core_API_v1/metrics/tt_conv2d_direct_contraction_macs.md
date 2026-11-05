# `tt_conv2d_direct_contraction_macs`

**Stability:** A
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

`tt_conv2d_forward` が各出力位置で行う `tt_linear_forward` に対応する、1画像あたりの理論MACsを返す。

## Signature

```python
tt_conv2d_direct_contraction_macs(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    ranks: Sequence[int],
    out_height: int,
    out_width: int,
) -> int
```

## 引数

- `out_modes`: 出力channelのphysical modes。
- `in_modes`: 入力patch幅のphysical modes。
- `ranks`: 境界を含むbond rank。両端は1。
- `out_height`: 実際の出力高さ。stride、padding、dilation、入力サイズから決めた値を渡す。
- `out_width`: 実際の出力幅。

## 戻り値

`out_height * out_width * sum(tt_linear_direct_contraction_macs(...))` の整数。

## 使用場面

学習済みTTConvのrank配分を、直接縮約forwardの理論MACsで比べるとき。

## 処理概要

1. 出力空間サイズを正の整数にする。
2. 既存のTTLinear直接縮約MACsをsiteごとに計算する。
3. 出力位置数を掛ける。rankの位置はsite別の式が区別する。

## 主なcontract / 注意事項

- unfold、bias、ReLU、pooling、backwardは含めない。
- dense kernel再構成でも、`compressed_conv2d_macs` の2層Conv分解でもない。
- mode、rank、出力サイズの不正値は `tt_linear_direct_contraction_macs` または正整数の検証と同じ例外にする。

## 関連API

`tt_linear_direct_contraction_macs`, `tt_conv2d_forward`, `conv2d_macs`
