# `estimate_mlp_macs`

**Stability:** B  
**定義:** `src/nn_compression/metrics/mlp_macs.py`

## 責務

現行784→512→256→10 MLPについて、`fc1/fc2`をSVD 2層化した場合のmodel-level理論MACsを比較する。

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

- `model`: 現行`fc1 / fc2 / fc3`を持つMLP。baseline各Linearのshapeを読み取る対象。
- `fc1_rank`: `fc1`をSVD 2層化するときの中間rank。
- `fc2_rank`: `fc2`をSVD 2層化するときの中間rank。
- `verbose`: `True`ならbaseline/compressed MACsと削減率を表示する。

## 戻り値

`(baseline_macs, compressed_macs, compute_reduction_rate)`。

- `baseline_macs`: 未圧縮`fc1 + fc2 + fc3`の理論MAC総数。
- `compressed_macs`: `fc1/fc2`だけを指定rankで2層化し、`fc3`は未圧縮のままにした理論MAC総数。
- `compute_reduction_rate`: baselineに対する理論MAC削減率。

## 使用場面

MNIST/Fashion-MNIST MLPの既存SVD実験。

## 処理概要

1. **baselineの3つのLinear MACsを計算する。**  
   `fc1 + fc2 + fc3`を`linear_macs()`で合計する。
2. **圧縮側`fc1`を2層Linearとして計算する。**  
   architecture既定値`784 → 512`と`fc1_rank`を使う。
3. **圧縮側`fc2`も2層Linearとして計算する。**  
   `512 → 256`と`fc2_rank`を使う。
4. **`fc3`は未圧縮なので元のMACsを加える。**
5. **baseline/compressedの合計から削減率を計算する。**
6. **必要なら表示してtupleを返す。**

## 主なcontract / 注意事項

generic MLP estimatorではなく現行`MNISTMLP` architecture前提のexperiment-support API。

## 関連API

`linear_macs`, `compressed_linear_macs`, `make_two_layer_svd_model`
