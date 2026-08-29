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

- `conv`: 計算量を比較する元Conv2d。baseline MACsの形状情報とcompressed MACsの分解元になる。
- `rank`: SVD 2層Convの中間channel数。
- `out_hw`: 対象Convの実際の出力feature map空間size`(H, W)`。MACsはこの出力位置数に比例する。
- `verbose`: `True`ならbaseline MACs、compressed MACs、削減率を表示する。

## 戻り値

`(baseline_macs, compressed_macs, compute_reduction)`の3要素tuple。

- `baseline_macs`: 元Conv2dの理論MAC総数。
- `compressed_macs`: 指定rankでSVD 2層化した場合の理論MAC総数。
- `compute_reduction`: `1 - compressed_macs / baseline_macs`で求める計算量削減率。正値が削減、負値なら計算量増加を表す。

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
