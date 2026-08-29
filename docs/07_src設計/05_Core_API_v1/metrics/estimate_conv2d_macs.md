# `estimate_conv2d_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

1つのConv2dについてbaseline/compressed MACsと削減率をまとめて計算する。

## Signature

```python
estimate_conv2d_macs(
    conv,
    rank: int,
    out_hw: tuple[int, int],
    verbose: bool = True,
) -> tuple[int, int, float]
```

## 引数

元Conv、rank、出力空間`(H, W)`、表示有無。

## 戻り値

```text
(baseline_macs, compressed_macs, compute_reduction)
```

## 使用場面

Notebookやrank sweepで1 Convの理論計算量を比較するとき。

## ざっくりした処理

`conv2d_macs()`と`compressed_conv2d_macs()`を計算し、`1 - compressed/baseline`を返す。

## 主なcontract / 注意事項

`out_hw`は呼び出し側が実際の出力shapeに合わせて渡す。

## 関連API

`conv2d_macs`, `compressed_conv2d_macs`, `sweep_conv2d_ranks`
