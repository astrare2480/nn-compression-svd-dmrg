# `retained_energy_from_matrix`

**Stability:** A  
**定義:** `src/nn_compression/compression/svd.py`

## 責務

上位`rank`個の特異値が、行列全体のFrobenius energyをどれだけ保持するか計算する。

## Signature

```python
retained_energy_from_matrix(
    matrix: torch.Tensor,
    rank: int,
) -> float
```

## 引数

- `matrix`: 2次元Tensor。
- `rank`: 評価するtruncated rank。

## 戻り値

```text
sum(S[:rank] ** 2) / sum(S ** 2)
```

のPython `float`。ゼロ行列では `1.0`。

## 使用場面

rank候補がweight情報をどの程度保持するか、task accuracyとは別の指標で確認するとき。

## ざっくりした処理

```text
matrix
→ rank validation
→ torch.linalg.svdvals
→ 特異値二乗和の比率
```

## 主なcontract / 注意事項

retained energyが高くてもtask accuracy維持を保証しない。

## 関連API

`retained_energy`, `truncated_svd`
