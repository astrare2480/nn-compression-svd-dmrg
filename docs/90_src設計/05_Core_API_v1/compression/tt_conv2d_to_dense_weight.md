# `tt_conv2d_to_dense_weight`

**Stability:** A
**定義:** `src/nn_compression/compression/tt_conv.py`

## 責務

TT-matrix core列から、診断・等価性検証・dense再変換用のConv2d weightを復元する。

## Signature
```python
tt_conv2d_to_dense_weight(cores, out_modes, in_modes, *, in_channels, kernel_size)
```

## 引数

- `cores`: Conv weight行列を表すTT-matrix core列。
- `out_modes`: 積が出力channel数となるmode列。
- `in_modes`: 積が`in_channels*kH*kW`となるmode列。
- `in_channels`: 復元後kernelの入力channel数。
- `kernel_size`: 復元後kernelの高さ・幅。

## 戻り値

shape `(out_channels, in_channels, kH, kW)`のTensor。coreのdtype/device/autogradを維持し、core storageを共有しない。

## 使用場面

direct TT-Convと`F.conv2d`の数値・勾配比較、診断表示、dense層への再変換。

## 処理概要

`tt_matrix_to_dense`でshape `(out_channels, in_channels*kH*kW)`へ復元し、Conv2d weight shapeへreshapeする。

## 主なcontract / 注意事項

- `TTConv2d.forward`からは呼ばない。production forwardはdense weightを作らない。
- mode積、channel、kernel sizeを検証する。`d=1`も対応する。

## 関連API

`tt_matrix_to_dense`, `tt_conv2d_forward`, `TTConv2d`
