# `tucker2_decompose_conv_weight`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

4階Conv2d weightのchannel mode 0（出力channel）/ 1（入力channel）だけをHOSVDし、Tucker-2のcoreとfactorを返す。

## Signature

```python
tucker2_decompose_conv_weight(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

## 引数

- `weight`: Tucker-2分解するConv2d weight。shapeは`(C_out, C_in, kH, kW)`で、channel mode 0/1だけを圧縮しkernel mode 2/3は維持する。
- `rank_out`: 出力channel mode（mode 0）に残すrank。`C_out`から`R_out`へ縮約するfactor sizeを決める。
- `rank_in`: 入力channel mode（mode 1）に残すrank。`C_in`から`R_in`へ縮約するfactor sizeを決める。

## 戻り値

HOSVD Tucker-2の3 componentを`(core, u_out, u_in)`の順で返す。

```text
core  : (rank_out, rank_in, kH, kW)  # channel方向を縮約した中央core
u_out : (C_out, rank_out)             # 出力channel方向factor
u_in  : (C_in, rank_in)               # 入力channel方向factor
```

この3つは`build_tucker2_conv_from_components()`へそのまま渡せる共通component形式。

## 使用場面

Moduleを作らずHOSVD Tucker-2 componentsだけ取得したいとき、HOOI分解結果と同じcomponent形式で比較するとき。

## 処理概要

1. **Conv weightが4階Tensorか確認する。**  
   Tucker-2 Convのmode意味を`(C_out, C_in, kH, kW)`へ固定する。
2. **channel rankを基本範囲で検証する。**  
   `rank_out <= C_out`, `rank_in <= C_in`を要求し、bool/Tensor scalar rankを拒否する。
3. **generic Tucker用のrank Mappingを作る。**  
   `{0: rank_out, 1: rank_in}`とし、空間mode 2/3は圧縮対象に含めない。
4. **`hosvd(weight, ranks)`へ分解を委譲する。**  
   ここで実数浮動小数点dtypeとmode-unfolding上の最大rankも検証される。
5. **generic factor辞書からchannel factorを取り出す。**  
   `factors[0]`を`u_out`、`factors[1]`を`u_in`とする。
6. **`core, u_out, u_in`の順で返す。**

## 主なcontract / 注意事項

分解rankはchannel数だけでなく、最終的にはgeneric HOSVDのmode-unfolding実現可能上限にも従う。

## 関連API

`hosvd`, `build_tucker2_conv`, `tucker2_hooi`
