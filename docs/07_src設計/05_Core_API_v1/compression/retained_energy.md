# `retained_energy`

**Stability:** A  
**定義:** `src/nn_compression/compression/svd.py`

## 責務

NN layerのweightをSVD用行列へ変換し、指定rankのretained energyを返す。

## Signature

```python
retained_energy(layer: nn.Module, rank: int) -> float
```

## 引数

- `layer`: `weight` を持つNN layer。
- `rank`: 評価rank。

## 戻り値

retained energyのPython `float`。

## 使用場面

Linear/Convのrank sweep結果へweight近似品質を付加するとき。

## ざっくりした処理

```text
layer.weight.detach()
→ (out, -1) に行列化
→ retained_energy_from_matrix
```

## 主なcontract / 注意事項

評価指標なので元weightへのautograd graphは追跡しない。

## 関連API

`retained_energy_from_matrix`, `sweep_layer_ranks`
