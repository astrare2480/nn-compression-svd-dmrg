# `get_endpoints`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

指定columnでsortした候補群の先頭点と末尾点を返す。

## Signature

```python
get_endpoints(df, column, x, y)
```

## 引数

DataFrame、sort列、x/y列名。

## 戻り値

```text
((left_x, left_y), (right_x, right_y))
```

## 使用場面

Pareto frontierの端点を結ぶknee基準直線を作るとき。

## ざっくりした処理

frame validation → sort → 先頭/末尾rowを座標tupleへ変換。

## 主なcontract / 注意事項

1点だけなら左右とも同じ候補になる。

## 関連API

`line_equation`, `find_knee_point`
