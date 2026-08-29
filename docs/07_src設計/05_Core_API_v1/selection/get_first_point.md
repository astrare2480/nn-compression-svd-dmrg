# `get_first_point`

**Stability:** B  
**定義:** `src/nn_compression/selection/knee.py`

## 責務

指定columnでDataFrameをsortし、その先頭候補を`(x, y)`座標として返す。

## Signature

```python
get_first_point(df, column, x, y)
```

## 引数

DataFrame、sortに使う列名、座標として読むx/y列名。

## 戻り値

`(float_x, float_y)`。

## 使用場面

knee計算の端点確認やhistorical Notebook互換処理。

## 処理概要

1. **入力DataFrameとx/y列を検証する。**  
   空DataFrame、列不足、NaN、infを拒否し、x/yを数値floatとして扱える形へ揃える。
2. **sort対象`column`が存在するか確認する。**
3. **`column`で昇順sortし、indexを振り直す。**
4. **先頭rowを取得する。**
5. **そのrowのx/yをPython floatへ変換してtupleで返す。**

## 主なcontract / 注意事項

入力が不正な場合に`None`等を返して続行せず、明示的にエラーにする。

## 関連API

`get_endpoints`, `line_equation`
