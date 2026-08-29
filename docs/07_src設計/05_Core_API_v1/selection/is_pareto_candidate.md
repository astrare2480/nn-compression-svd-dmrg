# `is_pareto_candidate`

**Stability:** B  
**定義:** `src/nn_compression/selection/pareto.py`

## 責務

DataFrame内の1候補が、他候補に2目的とも支配されていないか判定する。

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

- `df`: 候補全体。
- `row`: 判定対象行。
- `x_column`, `y_column`: 比較する2目的列。

## 戻り値

非支配なら`True`。

## 使用場面

parameter数とvalidation loss等、2目的のrank candidate選別。

## ざっくりした処理

```text
他候補が x <= row.x かつ y <= row.y
かつ少なくとも一方がstrictly smallerか判定
→ 1つでもあればdominated
```

## 主なcontract / 注意事項

2目的とも**小さいほど良い**前提。

## 関連API

`extract_pareto_frontier`
