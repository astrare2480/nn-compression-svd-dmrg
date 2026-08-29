# `conv2d_weight_matrix`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_svd.py`

## 責務

Conv2d weight `(C_out, C_in, kH, kW)` を、出力channelを行方向とするSVD用2次元行列 `(C_out, C_in*kH*kW)` へ変換する。

## Signature

```python
conv2d_weight_matrix(weight: torch.Tensor) -> torch.Tensor
```

## 引数

`weight`: Conv2dのweight Tensor。

## 戻り値

先頭dimensionを維持し、残りdimensionをflattenした2次元Tensor。

## 使用場面

Conv2d SVDの前処理、Conv weightのrank上限・特異値構造を考えるとき。

## 処理の流れ（日本語）

1. **出力channel軸をそのまま残す。**  
   Conv weightのaxis 0をSVD行列の行に対応させる。
2. **入力channelとkernel空間を1つの列dimensionへまとめる。**  
   `flatten(start_dim=1)`により`C_in * kH * kW`列の行列にする。
3. **2次元行列を返す。**  
   この関数ではSVDやrank validationは行わず、行列化だけを担当する。

### 処理フロー（短縮版）

```text
(C_out, C_in, kH, kW)
→ axis 0を維持
→ axis 1以降をflatten
→ (C_out, C_in*kH*kW)
```

## 主なcontract / 注意事項

このflatteningは通常の`groups=1` Conv2dを低rank2層へ分解する設計と対応する。

## 関連API

`factorize_conv2d_layer`, `truncated_svd`, `compressed_conv2d_macs`
