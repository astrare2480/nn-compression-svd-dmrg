# `tucker2_decompose_conv_weight`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

4階Conv2d weightのchannel mode 0/1だけをHOSVDし、Tucker-2 componentsを返す。

## Signature

```python
tucker2_decompose_conv_weight(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

## 引数

- `weight`: `(C_out, C_in, kH, kW)` の実数浮動小数点Tensor。
- `rank_out`: mode 0 rank。
- `rank_in`: mode 1 rank。

## 戻り値

```text
core  : (rank_out, rank_in, kH, kW)
u_out : (C_out, rank_out)
u_in  : (C_in, rank_in)
```

## 使用場面

Moduleを作らずHOSVD Tucker-2 componentsだけ取得したいとき、HOOI結果と同形式で比較するとき。

## ざっくりした処理

```text
4D/rank validation
→ ranks={0: rank_out, 1: rank_in}
→ hosvd(weight, ranks)
→ core, factor[0], factor[1]
```

## 主なcontract / 注意事項

分解rankはchannel数だけでなくmode-unfoldingの実現可能上限にも従う。

## 関連API

`hosvd`, `build_tucker2_conv`, `tucker2_hooi`
