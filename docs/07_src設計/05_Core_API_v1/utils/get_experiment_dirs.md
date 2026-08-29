# `get_experiment_dirs`

**Stability:** B  
**定義:** `src/nn_compression/utils/paths.py`

## 責務

実験成果物のdata/models/results directoryを共通規則で生成・取得する。

## Signature

```python
get_experiment_dirs(
    project_root: Path | str,
    method_name: str,
    case_name: str,
    experiment_name: str,
    *,
    create: bool = True,
) -> tuple[Path, Path, Path]
```

## 引数

project root、method/case/experiment名、directoryを作るか。

## 戻り値

`(data_dir, models_dir, results_dir)`。

## 使用場面

SVD/Tucker/TT/MPS/DMRG Notebookの保存先を同じ分類軸で揃えるとき。

## ざっくりした処理

```text
data = root/data
models = root/models/method/case/experiment
results = root/results/method/case/experiment
→ create=Trueならmkdir
```

## 主なcontract / 注意事項

Notebookと成果物で`method → case → experiment`の分類軸を揃える。

## 関連API

`find_project_root`
