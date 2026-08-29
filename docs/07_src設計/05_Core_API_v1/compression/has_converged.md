# `has_converged`

**Stability:** A  
**定義:** `src/nn_compression/compression/hooi.py`

## 責務

HOOIの前回誤差と現在誤差の差が、絶対・相対tolerance内か判定する。

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

- `error`: 現在の誤差。
- `prev_error`: 前回誤差。
- `abs_tol`: 絶対許容値。
- `rel_tol`: 前回誤差に比例する相対許容値。

## 戻り値

```python
abs(error - prev_error) <= abs_tol + rel_tol * abs(prev_error)
```

なら`True`。

## 使用場面

HOOI等の反復法の停止判定。

## ざっくりした処理

`tolerance validation → 誤差変化量計算 → 閾値比較`。

## 主なcontract / 注意事項

`tolerance`は有限・0以上・bool不可。

## 関連API

`hooi`
