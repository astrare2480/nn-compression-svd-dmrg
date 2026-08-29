# `estimate_cnn_macs`

**Stability:** B  
**定義:** `src/nn_compression/metrics/cnn_macs.py`

## 責務

指定したConv/Linear群を合計し、CNN全体のbaseline/compressed理論MACsを比較する。

## Signature

```python
estimate_cnn_macs(
    model,
    conv2_rank: int | None = None,
    fc1_rank: int | None = None,
    verbose: bool = True,
    *,
    conv_ranks: dict[str, int] | None = None,
    linear_ranks: dict[str, int] | None = None,
    conv_output_hw: dict[str, tuple[int, int]] | None = None,
    linear_layer_names: tuple[str, ...] = ("fc1", "fc2"),
) -> tuple[int, int, float]
```

## 引数

- positional `conv2_rank/fc1_rank`: Fashion-MNIST互換。
- `conv_ranks`: `{conv_name: rank}`。
- `linear_ranks`: `{linear_name: rank}`。
- `conv_output_hw`: `{conv_name: (H,W)}`。
- `linear_layer_names`: 集計対象Linear名。

## 戻り値

`(baseline_macs, compressed_macs, reduction)`。

## 使用場面

複数層SVD圧縮のmodel-level理論計算量比較。

## ざっくりした処理

```text
各Convをbaseline計算
→ rank指定層だけcompressed式へ置換
→ 各Linearも同様
→ 全層合計
→ reduction
```

## 主なcontract / 注意事項

default値はFashion-MNIST historical実験互換。CIFAR等ではnamed引数で対象層・空間sizeを明示する。

## 関連API

`conv2d_macs`, `compressed_conv2d_macs`, `linear_macs`, `compressed_linear_macs`
