# `tucker2_hooi_sweep`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

Conv weightのmode 0/1だけを対象に、Tucker-2 HOOIを1 sweep実行する。

## Signature

```python
tucker2_hooi_sweep(
    weight: torch.Tensor,
    factors: dict[int, torch.Tensor],
    rank_out: int,
    rank_in: int,
) -> dict[int, torch.Tensor]
```

## 引数

- `weight`: 4階Conv weight。
- `factors`: 現在のmode 0/1 factors。
- `rank_out`, `rank_in`: channel ranks。

## 戻り値

更新されたfactor dict。

## 使用場面

Tucker-2 HOOIの更新過程をsweep単位で確認するとき。

## ざっくりした処理

```text
rank validation
→ ranks={0: rank_out, 1: rank_in}
→ generic hooi_sweepへ委譲
```

## 主なcontract / 注意事項

空間mode 2/3は`ranks`へ含めないため圧縮しない。

## 関連API

`hooi_sweep`, `tucker2_hooi`
