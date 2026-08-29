# `get_endpoints`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

指定columnでsortした候補群の先頭点と末尾点を`(x, y)`座標で返す。

## Signature

```python
get_endpoints(df, column, x, y)
```

## 引数

DataFrame、sort列、座標として読むx/y列名。

## 戻り値

```text
((left_x, left_y), (right_x, right_y))
```

## 使用場面

Pareto frontierの両端を結ぶ直線を作り、knee距離を測るとき。

## 処理概要

1. **DataFrameとx/y列を共通helperで検証する。**  
   空、NaN、infを拒否する。
2. **sort列が存在するか確認する。**
3. **指定columnで昇順sortする。**
4. **先頭rowをleft endpointとして取得する。**
5. **末尾rowをright endpointとして取得する。**
6. **両rowのx/yをfloatへ変換して2点tupleとして返す。**

## 主なcontract / 注意事項

候補が1点だけならleft/rightは同じ点になる。この場合は後続`line_equation()`が退化線として扱う。

## 関連API

`line_equation`, `find_knee_point`
