# `build_tucker2_conv_from_components`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

既に計算済みのTucker-2 core/factorsから、等価な3層Conv Moduleを構築する。

## Signature

```python
build_tucker2_conv_from_components(
    conv: nn.Conv2d,
    core: torch.Tensor,
    u_out: torch.Tensor,
    u_in: torch.Tensor,
) -> nn.Sequential
```

## 引数

- `conv`: 元の`groups=1` Conv2d。shape/stride/padding/bias等の基準。
- `core`: `(R_out, R_in, kH, kW)`。
- `u_out`: `(C_out, R_out)`。
- `u_in`: `(C_in, R_in)`。

## 戻り値

```text
1x1 Conv(C_in → R_in, bias=False)
→ core Conv(R_in → R_out, original spatial config, bias=False)
→ 1x1 Conv(R_out → C_out, bias=original)
```

## 使用場面

HOSVD/HOOIなど分解法を問わず、共通のTucker-2 Moduleへ変換したいとき。

## ざっくりした処理

```text
Conv/components validation
→ 3層Convを元device/dtypeで生成
→ U_in.T/core/U_outをcopy
→ biasを出力projectionへcopy
→ requires_grad継承
```

## 主なcontract / 注意事項

- componentsのshape/dtype/deviceを厳密に確認。
- 新Parameterはleafで元weight storageを共有しない。
- これは分解APIではないためmode-unfolding rank feasibilityを再計算しない。

## 関連API

`build_tucker2_conv`, `tucker2_decompose_conv_weight`, `tucker2_hooi`
