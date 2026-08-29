# `find_knee_point_numpy`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

`find_knee_point()`と同じknee判定を、NumPy配列のvectorized計算で行う。

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

## 引数

- `df`: knee候補をrowとして持つDataFrame。x/y列をNumPy arrayへ変換して一括計算する。
- `slope`: `line_equation()`で作った基準直線の傾き。垂直線では`math.inf`。
- `intercept`: 通常線ではy切片、垂直線では一定x座標。
- `x`: candidateのx座標として使う列名。
- `y`: candidateのy座標として使う列名。

## 戻り値

`((knee_x, knee_y), distance)`。意味はloop版`find_knee_point()`と同じで、最大距離candidateの座標と基準直線までの距離をPython floatで返す。

## 使用場面

同じknee semanticsをPython loopではなく配列演算でまとめて計算したいとき。

## 処理概要

1. **DataFrameを検証し、x/y列をfloat NumPy arrayへ変換する。**
2. **直線パラメータを検証する。**
3. **候補が1点ならその点・距離0を返す。**
4. **垂直線か通常線かでvectorized距離式を選ぶ。**
5. **全候補の距離arrayを一括計算する。**
6. **距離arrayにNaN/infがないか確認する。**
7. **`np.argmax()`で最大距離indexを求める。**
8. **その座標と距離をPython floatへ変換して返す。**

## 主なcontract / 注意事項

loop版と同じ意味を維持することを優先する。実装方式が違ってもknee semanticsは変えない。

## 関連API

`find_knee_point`, `line_equation`
