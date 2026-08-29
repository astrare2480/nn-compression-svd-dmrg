# `extract_pareto_frontier`

**Stability:** B  
**定義:** `src/nn_compression/selection/pareto.py`

## 責務

2目的で非支配な候補だけをDataFrameから抽出する。

## Signature

```python
extract_pareto_frontier(
    df,
    x_column,
    y_column,
)
```

## 引数

候補DataFrameと、最小化する2列名。

## 戻り値

非支配候補だけを`x_column`昇順へ並べた新しいDataFrame。

## 使用場面

rank sweep結果から、明らかに劣る候補を除外するとき。

## ざっくりした処理

各rowへ`is_pareto_candidate()`を適用 → 非支配indexを抽出 → x順sort → index reset。

## 主なcontract / 注意事項

maximize目的を自動変換しない。

## 関連API

`is_pareto_candidate`, `find_knee_point`
