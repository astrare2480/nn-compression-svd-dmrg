# `find_knee_point_numpy`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

`find_knee_point()`と同じknee判定をNumPy vectorized計算で行う。

## Signature

```python
find_knee_point_numpy(
    df,
    slope,
    intercept,
    x,
    y,
)
```

## 引数 / 戻り値

意味は`find_knee_point()`と同じ。

## 使用場面

同一knee semanticsをNumPy配列でまとめて計算したいとき。

## ざっくりした処理

x/y列をNumPy配列化 → 点-直線距離をvectorized計算 → `argmax`でkneeを選択。

## 主なcontract / 注意事項

loop版と同じ意味を持つことを優先する。NaN/inf距離は拒否。

## 関連API

`find_knee_point`, `line_equation`
