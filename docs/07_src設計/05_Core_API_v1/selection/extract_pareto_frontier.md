# `extract_pareto_frontier`

**Stability:** B  
**定義:** `src/nn_compression/selection/pareto.py`

## 責務

2目的で非支配な候補だけをDataFrameから抽出し、Pareto frontierとして返す。

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

非支配候補だけを`x_column`昇順へ並べ、indexを振り直した新しいDataFrame。

## 使用場面

rank sweep結果から「両目的で他候補より明らかに劣る点」を除外するとき。

## 処理概要

1. **保持するrow index用listを作る。**
2. **DataFrameを1行ずつ走査する。**
3. **各rowへ`is_pareto_candidate()`を適用する。**  
   他候補に支配されていないかを2目的で判定する。
4. **非支配なら元indexを保持listへ追加する。**
5. **保持indexだけを元DataFrameからcopyする。**  
   元DataFrame自体は変更しない。
6. **`x_column`の昇順へsortする。**
7. **indexを0から振り直して返す。**

## 主なcontract / 注意事項

maximize目的を自動変換しない。x/yとも小さいほど良い値へ事前に揃える。

## 関連API

`is_pareto_candidate`, `find_knee_point`
