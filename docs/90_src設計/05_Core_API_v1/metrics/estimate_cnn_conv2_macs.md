# `estimate_cnn_conv2_macs`

**Stability:** B  
**定義:** `src/nn_compression/metrics/cnn_macs.py`

## 責務

model内の指定named Conv2d 1層について、baseline/compressed MACsと削減率を計算する。

## Signature

```python
estimate_cnn_conv2_macs(
    model,
    rank,
    verbose=True,
    *,
    layer_name: str = "conv2",
    out_hw: tuple[int, int] | None = None,
)
```

## 引数

- `model`: MACsを評価するnamed Conv2dを含むmodel。
- `rank`: 対象ConvをSVD 2層化するときの中間channel rank。
- `verbose`: `True`ならbaseline/compressed MACsと削減率を表示する。
- `layer_name`: model内で評価するConv2dのnamed path。既定`"conv2"`はhistorical Fashion-MNIST実験互換。
- `out_hw`: 対象Convの実際の出力feature map size`(H, W)`。`None`ならhistorical `conv2`用の`(14, 14)`を使う。

## 戻り値

`(baseline_macs, compressed_macs, compute_reduction)`。

- `baseline_macs`: 指定named Conv2dの未圧縮MACs。
- `compressed_macs`: 指定rankで2層化した場合のMACs。
- `compute_reduction`: baselineに対する理論MAC削減率。

## 使用場面

Fashion-MNIST conv2、または任意named Conv 1層の理論MACs比較。

## 処理概要

1. **`out_hw`省略時はhistorical default `(14,14)`を採用する。**
2. **`get_named_module(model, layer_name)`で対象Convを取得する。**
3. **取得したConv、rank、out_hwを`estimate_conv2d_macs()`へ渡す。**
4. **共通Conv MACs APIの3要素tupleをそのまま返す。**

## 主なcontract / 注意事項

任意Convで使う場合は実際の`out_hw`を明示する。defaultはFashion-MNIST互換であり自動shape推論ではない。

## 関連API

`estimate_conv2d_macs`, `estimate_cnn_macs`
