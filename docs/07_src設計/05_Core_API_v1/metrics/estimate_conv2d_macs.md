# `estimate_conv2d_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

1つのConv2dについてbaseline MACs、SVD 2層化後MACs、削減率をまとめて計算する。

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

`(baseline_macs, compressed_macs, compute_reduction)`。

## 使用場面

Notebookやrank sweepで1 Convの理論計算量を比較するとき。

## 処理概要

1. **圧縮対象として`groups=1` Convか確認する。**
2. **`out_hw`を`out_h`, `out_w`へ分ける。**
3. **`conv2d_macs()`で元ConvのMACsを計算する。**
4. **`compressed_conv2d_macs()`でSVD 2層ConvのMACsを計算する。**
5. **`1 - compressed / baseline`で削減率を求める。**
6. **`verbose=True`なら3つの値を表示する。**
7. **3要素tupleを返す。**

## 主なcontract / 注意事項

`out_hw`は呼び出し側が対象Convの実際の出力shapeに合わせて渡す。

## 関連API

`conv2d_macs`, `compressed_conv2d_macs`, `sweep_conv2d_ranks`
