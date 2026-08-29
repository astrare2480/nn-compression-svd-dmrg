# `line_equation`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

2点を通る直線を、後続の点-直線距離計算で扱える `(slope, intercept)` 形式へ変換する。

## Signature

```python
line_equation(p0, p1)
```

## 引数

`p0`, `p1`: `(x, y)`座標。

## 戻り値

- 通常線: `(slope, y_intercept)`。
- 垂直線: `(math.inf, x_constant)`。
- 完全に同じ点: `(0.0, y0)`。

## 使用場面

Pareto frontierの端点を結ぶ基準直線を作り、knee candidateとの距離を測るとき。

## 処理概要

1. **2点のx/yをfloatへ変換する。**
2. **4つの座標が有限値か確認する。**
3. **2点が完全に同じか確認する。**  
   同一点なら通常の傾き計算ができないため、水平な退化線として`(0.0, y0)`を返す。
4. **x座標だけが同じか確認する。**  
   垂直線の場合は傾きを`inf`、第2戻り値へ`x_constant`を格納する。
5. **通常線なら傾き `(y1-y0)/(x1-x0)` を計算する。**
6. **`y = slope*x + intercept`を満たす切片を計算する。**
7. **直線表現を返す。**

## 主なcontract / 注意事項

垂直線では第2戻り値がy切片ではなく`x_constant`。後続距離計算もこの規約を理解している。

## 関連API

`get_endpoints`, `find_knee_point`, `find_knee_point_numpy`
