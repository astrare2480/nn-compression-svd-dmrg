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

候補DataFrame、`line_equation()`の直線表現、x/y列名。

## 戻り値

```text
((knee_x, knee_y), distance)
```

## 使用場面

Pareto frontierから圧縮量と性能の折衷点となる代表candidateを選ぶ補助指標。

## 処理の流れ（日本語）

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

### 処理フロー（短縮版）

```text
candidate frame検証
→ line検証
→ 各pointのline distance
→ 最大値を逐次更新
→ knee point + distance
```

## 主なcontract / 注意事項

- 1点ならその点・距離0。
- 全点が直線上なら先頭の最大距離点（距離0）を返す。
- `(None, None)`は返さない。

## 関連API

`line_equation`, `find_knee_point_numpy`
