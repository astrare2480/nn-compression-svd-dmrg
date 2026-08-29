# `find_knee_point`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

各候補から端点直線までの距離を計算し、最も遠い点をkneeとして返すloop版。

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

Pareto frontierから代表rank候補を選ぶ補助指標。

## ざっくりした処理

入力検証 → 各rowの点-直線距離 → 最大距離点を保持。

## 主なcontract / 注意事項

- 1点ならその点・距離0。
- 全点が直線上なら先頭の最大距離点（距離0）を返す。
- `(None, None)`は返さない。

## 関連API

`line_equation`, `find_knee_point_numpy`
