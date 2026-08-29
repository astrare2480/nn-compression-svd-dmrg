# `find_project_root`

**Stability:** B  
**定義:** `src/nn_compression/utils/paths.py`

## 責務

開始pathから上位directoryを辿り、`.git`が存在するproject rootを見つける。

## Signature

```python
find_project_root(
    start_path: Path | str,
) -> Path
```

## 引数

探索開始path。

## 戻り値

project rootの`Path`。

## 使用場面

Notebookのcwdに依存せずdata/models/resultsをproject基準で解決するとき。

## ざっくりした処理

`start_path.resolve()` → 自身とparentsを順に確認 → `.git`があれば返す。

## 主なcontract / 注意事項

`.git`のないzip展開物等では`FileNotFoundError`。

## 関連API

`get_experiment_dirs`
