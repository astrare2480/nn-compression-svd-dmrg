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

model、rank、対象layer名、出力空間size。`out_hw=None`ならhistorical Fashion-MNIST `conv2`の14×14を使う。

## 戻り値

`estimate_conv2d_macs()`と同じ`(baseline, compressed, reduction)`。

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
