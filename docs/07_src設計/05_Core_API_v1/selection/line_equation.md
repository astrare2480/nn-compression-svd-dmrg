# `line_equation`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

2点を通る直線を、後続knee距離計算で使う `(slope, intercept)` 形式へ変換する。

## Signature

```python
line_equation(p0, p1)
```

## 引数

`p0`, `p1`: `(x, y)`座標。

## 戻り値

- 通常線: `(slope, y_intercept)`。
- 垂直線: `(inf, x_constant)`。
- 同一点: `(0.0, y0)`。

## 使用場面

Pareto frontier端点を結ぶ直線からknee距離を測るとき。

## ざっくりした処理

有限値確認 → 同一点/垂直線を特殊処理 → 通常は傾きと切片を計算。

## 主なcontract / 注意事項

垂直線では第2戻り値がy切片ではなく`x_constant`になる。

## 関連API

`get_endpoints`, `find_knee_point`, `find_knee_point_numpy`
