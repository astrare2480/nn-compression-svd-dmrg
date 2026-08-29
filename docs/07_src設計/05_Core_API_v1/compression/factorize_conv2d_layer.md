# `factorize_conv2d_layer`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_svd.py`

## 責務

1つのConv2dをSVDで2層Convへ分解する。

## Signature

```python
factorize_conv2d_layer(
    conv: nn.Conv2d,
    rank: int,
) -> nn.Sequential
```

## 引数

- `conv`: `groups=1` の通常Conv2d。
- `rank`: 中間channel数。

## 戻り値

```text
Conv2d(C_in → rank, 元kernel, bias=False)
→ Conv2d(rank → C_out, 1x1, bias=original)
```

## 使用場面

Conv2dのSVD低rank圧縮。

## ざっくりした処理

```text
weightを(out, in*kH*kW)へflatten
→ truncated_svd
→ Vh_rを空間Convへ配置
→ U_r @ diag(S_r)を1x1 Convへ配置
→ biasを出力側へcopy
```

## 主なcontract / 注意事項

- `groups=1`、非transposed Convのみ。
- rank上限 `min(C_out, C_in*kH*kW)`。
- 元stride/padding/dilation/padding_modeは1層目へ継承。
- device/dtype/requires_grad維持。

## 関連API

`conv2d_weight_matrix`, `factorize_named_conv2d`, `compressed_conv2d_macs`
