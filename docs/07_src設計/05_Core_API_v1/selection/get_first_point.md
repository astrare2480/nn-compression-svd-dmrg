# `get_first_point`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

指定columnでDataFrameをsortし、先頭候補の `(x, y)` 座標を返す。

## Signature

```python
get_first_point(df, column, x, y)
```

## 引数

DataFrame、sort列名、座標として読むx/y列名。

## 戻り値

`(float_x, float_y)`。

## 使用場面

knee計算の端点確認やhistorical Notebook互換処理。

## ざっくりした処理

入力frameを検証 → `column`でsort → 先頭rowのx/yをfloat化。

## 主なcontract / 注意事項

空・NaN・inf・列不足を明示的に拒否する。

## 関連API

`get_endpoints`, `line_equation`
