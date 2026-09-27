# `find_knee_point`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

各候補から基準直線までの垂直距離を計算し、最も遠い点をkneeとして返すloop版実装。

## Signature

```python
find_knee_point(
    df,
    slope,
    intercept,
    x,
    y,
)
```

## 引数

- `df`: knee候補をrowとして持つDataFrame。通常はPareto frontierを渡す。
- `slope`: `line_equation()`で作った基準直線の傾き。垂直線の場合は`math.inf`。
- `intercept`: 通常線ではy切片、`slope == inf`の垂直線では一定x座標を表す。
- `x`: 各候補のx座標として読むDataFrame列名。
- `y`: 各候補のy座標として読むDataFrame列名。

## 戻り値

`((knee_x, knee_y), distance)`のtuple。

- `(knee_x, knee_y)`: 基準直線から最も離れたcandidateの座標。
- `distance`: そのcandidateから基準直線までの垂直距離。

候補が1点だけならその点と距離`0.0`を返す。

## 使用場面

Pareto frontierから圧縮量と性能の折衷点となる代表candidateを選ぶ補助指標。

## 処理概要

1. **DataFrameとx/yを検証・数値化する。**
2. **直線パラメータが有効か確認する。**  
   通常線または規約上の垂直線`inf`だけを許可する。
3. **候補が1点だけならその点を距離0で返す。**
4. **最大距離の初期値を-1にする。**  
   全点が直線上で距離0でも、最初の点を必ず選べるようにする。
5. **各rowのx/yから基準直線までの距離を計算する。**  
   通常線と垂直線で距離式を切り替える。
6. **現在の最大距離より大きければknee候補を更新する。**
7. **全候補走査後、最大距離点と距離を返す。**

## 主なcontract / 注意事項

- 1点ならその点・距離0。
- 全点が直線上なら先頭の最大距離点（距離0）を返す。
- `(None, None)`は返さない。

## 関連API

`line_equation`, `find_knee_point_numpy`
