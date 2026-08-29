# `is_pareto_candidate`

**Stability:** B  
**定義:** `src/nn_compression/selection/pareto.py`

## 責務

DataFrame内の1候補が、他候補に2目的とも支配されていないPareto候補か判定する。

## Signature

```python
is_pareto_candidate(
    df,
    row,
    x_column,
    y_column,
) -> bool
```

## 引数

- `df`: 候補全体を持つDataFrame。
- `row`: 判定対象の1行。
- `x_column`, `y_column`: 最小化対象の2目的列。

## 戻り値

他候補に支配されていなければ`True`、支配されていれば`False`。

## 使用場面

parameter数とvalidation lossなど、2つの「小さいほど良い」目的でrank candidateを絞るとき。

## 処理の流れ（日本語）

1. **判定対象`row`のx/y値を基準にする。**
2. **全候補について「xがrow以下」かを判定する。**
3. **同時に「yがrow以下」かを判定する。**
4. **さらに少なくとも一方がrowより厳密に小さい候補を探す。**  
   両方同値の点は「支配」とみなさない。
5. **3条件をすべて満たす候補が1つでもあればrowはdominatedと判定する。**
6. **dominatedの否定を返す。**

### 処理フロー（短縮版）

```text
row
→ 他候補で x <= row.x
   かつ y <= row.y
   かつ少なくとも一方 < row
→ 存在する: False
→ 存在しない: True
```

## 主なcontract / 注意事項

2目的とも**小さいほど良い**前提。maximize目的を自動で符号反転しない。

## 関連API

`extract_pareto_frontier`
