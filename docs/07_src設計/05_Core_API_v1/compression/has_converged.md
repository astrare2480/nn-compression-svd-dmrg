# `has_converged`

**Stability:** A  
**定義:** `src/nn_compression/compression/hooi.py`

## 責務

HOOIの前回誤差と現在誤差の変化が、絶対・相対toleranceの組合せで十分小さくなったか判定する。

## Signature

```python
has_converged(
    error: float,
    prev_error: float,
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> bool
```

## 引数

- `error`: 現在の再構成誤差。
- `prev_error`: 1回前の再構成誤差。
- `abs_tol`: 誤差の絶対変化に対する許容値。
- `rel_tol`: 前回誤差の大きさに比例する相対許容値。

## 戻り値

次を満たせば`True`。

```python
abs(error - prev_error) <= abs_tol + rel_tol * abs(prev_error)
```

## 使用場面

HOOIなど、誤差の改善量を見て反復を止める処理。

## 処理概要

1. **`abs_tol`を検証する。**  
   boolではない有限数で、0以上であることを要求する。
2. **`rel_tol`を同様に検証する。**
3. **現在誤差と前回誤差の絶対変化量を計算する。**  
   `absolute_change = abs(error - prev_error)`。
4. **許容閾値を計算する。**  
   `abs_tol + rel_tol * abs(prev_error)`として、絶対許容とスケール依存の相対許容を足す。
5. **変化量が閾値以下かをboolで返す。**

## 主なcontract / 注意事項

収束判定は「誤差そのものが小さいか」ではなく、**前回からの変化が小さいか**を見ている。

## 関連API

`hooi`
