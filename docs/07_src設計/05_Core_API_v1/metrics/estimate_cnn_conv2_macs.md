# `estimate_cnn_conv2_macs`

**Stability:** B  
**定義:** `src/nn_compression/metrics/cnn_macs.py`

## 責務

指定named Conv2d 1層について、baseline/compressed MACsを比較する。

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

model、rank、対象layer名、出力空間size。`out_hw=None`ならhistorical conv2の14×14。

## 戻り値

`estimate_conv2d_macs()`と同じ3要素tuple。

## 使用場面

Fashion-MNIST conv2や、named Conv 1層のMACs比較。

## ざっくりした処理

`get_named_module()`で対象Convを取り、`estimate_conv2d_macs()`へ委譲する。

## 主なcontract / 注意事項

任意Convで使う場合は実際の`out_hw`を明示する。

## 関連API

`estimate_conv2d_macs`, `estimate_cnn_macs`
