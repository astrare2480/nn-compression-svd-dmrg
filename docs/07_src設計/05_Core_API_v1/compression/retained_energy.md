# `retained_energy`

**Stability:** A  
**定義:** `src/nn_compression/compression/svd.py`

## 責務

NN layerのweightをSVD用の2次元行列へ変換し、指定rankで保持される特異値energy比率を返す。

## Signature

```python
retained_energy(layer: nn.Module, rank: int) -> float
```

## 引数

- `layer`: `weight`を持つNN layer。
- `rank`: retained energyを評価するrank。

## 戻り値

retained energyのPython `float`。

## 使用場面

Linear/Convのrank sweep結果へ、parameter数やaccuracyとは別にweight近似品質を追加するとき。

## 処理概要

1. **layerのweightを取得する。**  
   Parameterそのものではなく評価用の値として扱う。
2. **weightをautograd graphから切り離す。**  
   `layer.weight.detach()`を使い、retained energy計算が学習graphへ接続しないようにする。
3. **weightをSVD用の2次元行列へ変換する。**  
   Linear weightはそのまま、Conv等の2階を超えるweightは`(out, -1)`へreshapeする。
4. **行列版の共通関数へ委譲する。**  
   `retained_energy_from_matrix(matrix, rank)`で特異値二乗和の比率を計算する。
5. **得られたfloatを返す。**  
   layer固有の追加処理は行わない。

## 主なcontract / 注意事項

- 評価指標なので元weightへのautograd graphは追跡しない。
- retained energyはtask accuracyそのものではない。

## 関連API

`retained_energy_from_matrix`, `sweep_layer_ranks`
