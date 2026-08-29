# `build_tucker2_conv`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

1つのConv2dをHOSVD Tucker-2分解し、そのまま3層Conv `nn.Sequential`へ変換する。

## Signature

```python
build_tucker2_conv(
    conv: nn.Conv2d,
    rank_out: int,
    rank_in: int,
) -> nn.Sequential
```

## 引数

- `conv`: `groups=1`の通常Conv2d。
- `rank_out`, `rank_in`: channel mode rank。

## 戻り値

Tucker-2 3層Conv `nn.Sequential`。

## 使用場面

HOSVDでConv layerを直接Tucker-2へ置換したいとき。

## ざっくりした処理

```text
Conv semantic validation
→ conv.weight.detach()
→ tucker2_decompose_conv_weight
→ build_tucker2_conv_from_components
```

## 主なcontract / 注意事項

- Module構築は**autograd境界**。元weightのgraphは引き継がない。
- 構築後の新Parameterはleafで通常Fine-tuning可能。

## 関連API

`tucker2_decompose_conv_weight`, `build_tucker2_conv_from_components`, `tucker2_hooi`
