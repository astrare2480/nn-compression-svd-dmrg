# `estimate_mlp_macs`

**Stability:** B  
**定義:** `src/nn_compression/metrics/mlp_macs.py`

## 責務

現行784→512→256→10 MLPについて、fc1/fc2をSVD 2層化した場合のmodel-level MACsを比較する。

## Signature

```python
estimate_mlp_macs(
    model,
    fc1_rank,
    fc2_rank,
    verbose=True,
)
```

## 引数

現行MLP、fc1/fc2 rank、表示有無。

## 戻り値

`(baseline_macs, compressed_macs, compute_reduction_rate)`。

## 使用場面

MNIST/Fashion-MNIST MLP historical実験。

## ざっくりした処理

```text
baseline = fc1 + fc2 + fc3 MACs
compressed = factorized fc1 + factorized fc2 + original fc3
→ 削減率
```

## 主なcontract / 注意事項

generic MLP estimatorではなく現行architecture前提。

## 関連API

`linear_macs`, `compressed_linear_macs`, `make_two_layer_svd_model`
