# `retained_energy_from_matrix`

**Stability:** A  
**定義:** `src/nn_compression/compression/svd.py`

## 責務

上位`rank`個の特異値が、行列全体のFrobenius energy（特異値二乗和）をどれだけ保持するか計算する。

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

rank候補がweightのFrobenius norm上の情報量をどの程度保持するか確認するとき。task accuracyとは別のweight近似指標として使う。

## 処理の流れ（日本語）

1. **行列の最大rankを求める。**  
   `matrix`が2次元であることを確認し、`min(matrix.shape)`を最大rankとする。
2. **指定rankを検証する。**  
   `truncated_svd()`と同じ整数・範囲contractを使い、不可能なrankを拒否する。
3. **特異値だけを計算する。**  
   factor行列は不要なので`torch.linalg.svdvals(matrix)`を使い、特異値`S`だけを取得する。
4. **行列全体のenergyを計算する。**  
   `sum(S**2)`を分母とする。これはFrobenius normの二乗に対応する。
5. **ゼロ行列を特別扱いする。**  
   energyが0なら、どのrankでも失われるenergyがないという実装上の定義で`1.0`を返す。
6. **上位rankのenergy比率を計算する。**  
   `sum(S[:rank]**2) / sum(S**2)`をPython floatとして返す。

### 処理フロー（短縮版）

```text
matrix
→ 2D / rank検証
→ svdvals
→ 全特異値の二乗和
→ 上位rankの二乗和
→ 比率を返す
```

## 主なcontract / 注意事項

retained energyが高くてもtask accuracy維持を保証しない。weight近似とtask性能は別指標として扱う。

## 関連API

`retained_energy`, `truncated_svd`
